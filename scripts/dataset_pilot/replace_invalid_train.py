"""Replace invalid Train annotations with unseen, deduplicated source rows.

The replacement keeps the removed ``use_tool`` strata exactly, blocks every
row/group/image that has ever appeared in the frozen dataset, materializes new
assets from the pinned Arrow cache, and leaves the new rows unannotated for the
normal annotation command to process.
"""
import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image, ImageOps

from .common import read_jsonl, write_json, write_jsonl
from .prepare_formal import (dhash, image_hash, near_duplicate,
                             representative_groups, stable_score,
                             stream_digest, use_tool_flag)


def load_train_candidates(source_root: Path, metadata: dict) -> list[dict]:
    candidates = []
    offset = 0
    for shard in metadata["smart_train"]["files"]:
        path = source_root / "scan_index" / "smart_train" / f"{shard}.jsonl"
        meta_path = source_root / "scan_index" / "smart_train" / f"{shard}.meta.json"
        meta = json.loads(meta_path.read_text())
        for row in read_jsonl(path):
            candidates.append({**row, "source_row_id": offset + row["source_row_in_shard"]})
        offset += meta["row_count"]
    return candidates


def select_replacements(candidates: list[dict], targets: dict[bool, int], seed: int,
                        blocked_groups: set[str], blocked_pixels: set[str],
                        blocked_dhashes: list[str]) -> tuple[list[dict], Counter]:
    representatives, _ = representative_groups(candidates, 20260921, "smart_train")
    ordered = sorted(representatives,
                     key=lambda row: stable_score(seed, "train-replacement", row["group_key"]))
    selected = []
    counts = Counter()
    rejected = Counter()
    occupied_groups = set(blocked_groups)
    occupied_pixels = set(blocked_pixels)
    occupied_dhashes = list(blocked_dhashes)
    for row in ordered:
        stratum = use_tool_flag(row)
        if counts[stratum] >= targets.get(stratum, 0):
            continue
        if row["group_key"] in occupied_groups:
            rejected["group_overlap_or_previously_selected"] += 1
            continue
        if row["pixel_sha256"] in occupied_pixels:
            rejected["exact_duplicate_image"] += 1
            continue
        if near_duplicate(row["dhash"], occupied_dhashes):
            rejected["near_duplicate_image"] += 1
            continue
        selected.append({
            **row,
            "split": "train",
            "selection_seed": seed,
            "selection_score": stable_score(
                seed, "train-replacement", row["group_key"]),
        })
        counts[stratum] += 1
        occupied_groups.add(row["group_key"])
        occupied_pixels.add(row["pixel_sha256"])
        occupied_dhashes.append(row["dhash"])
        if all(counts[key] == value for key, value in targets.items()):
            break
    if any(counts[key] != value for key, value in targets.items()):
        raise RuntimeError(f"Replacement strata {dict(counts)} do not meet {targets}")
    return selected, rejected


def pair_replacements(invalid: list[dict], selected: list[dict], seed: int) -> list[dict]:
    paired = []
    for stratum in (False, True):
        removed = sorted((row for row in invalid if use_tool_flag(row)),
                         key=lambda row: row["sample_id"])
        if not stratum:
            removed = sorted((row for row in invalid if not use_tool_flag(row)),
                             key=lambda row: row["sample_id"])
        added = [row for row in selected if use_tool_flag(row) is stratum]
        if len(removed) != len(added):
            raise RuntimeError("Replacement pairing changed a use_tool stratum")
        for old, new in zip(removed, added):
            paired.append({
                **new,
                "replacement_meta": {
                    "batch_seed": seed,
                    "replaces_sample_id": old["sample_id"],
                    "reason": "invalid_annotation",
                },
            })
    return paired


def materialize(dataset: Path, source_root: Path, selected: list[dict]) -> list[dict]:
    wanted = defaultdict(dict)
    for row in selected:
        wanted[row["source_shard"]][row["source_row_in_shard"]] = row
    manifests = []
    for shard, shard_rows in sorted(wanted.items()):
        path = source_root / "source_cache" / "smart_train" / shard
        seen = set()
        row_number = 0
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                images = batch.column(batch.schema.get_field_index("images"))
                for index in range(batch.num_rows):
                    local_row = row_number
                    row_number += 1
                    if local_row not in shard_rows:
                        continue
                    selected_row = shard_rows[local_row]
                    blob = images[index].as_py()["bytes"]
                    if hashlib.sha256(blob).hexdigest() != selected_row["source_bytes_sha256"]:
                        raise ValueError("Source bytes changed between scan and replacement")
                    image = Image.open(io.BytesIO(blob))
                    image.load()
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    if image_hash(image) != selected_row["pixel_sha256"]:
                        raise ValueError("Pixels changed between scan and replacement")
                    sample_id = f"smart-train-{selected_row['source_row_id']:05d}"
                    image_path = Path("images") / f"{sample_id}.png"
                    original_path = Path("originals") / f"{sample_id}.bin"
                    lowres_path = Path("lowres") / f"{sample_id}.png"
                    for relative in (image_path, original_path, lowres_path):
                        (dataset / "train" / relative).parent.mkdir(parents=True, exist_ok=True)
                    (dataset / "train" / original_path).write_bytes(blob)
                    image.save(dataset / "train" / image_path)
                    lowres = image.copy()
                    lowres.thumbnail((selected_row["source_info"]["tgt_width"],
                                      selected_row["source_info"]["tgt_height"]),
                                     Image.Resampling.LANCZOS)
                    lowres.save(dataset / "train" / lowres_path)
                    manifests.append({
                        "sample_id": sample_id,
                        "source": selected_row["source"],
                        "source_revision": selected_row["source_revision"],
                        "source_access": "pinned full Arrow shard; original asset bytes retained",
                        "source_shard": shard,
                        "source_row_in_shard": selected_row["source_row_in_shard"],
                        "source_row_id": selected_row["source_row_id"],
                        "doc_id": selected_row["doc_id"],
                        "group_id": selected_row["group_key"],
                        "split": "train",
                        "formal_split_audit_pending": False,
                        "selection_seed": selected_row["selection_seed"],
                        "selection_score": selected_row["selection_score"],
                        "replacement_meta": selected_row["replacement_meta"],
                        "data_source": selected_row["data_source"],
                        "source_info": selected_row["source_info"],
                        "image_path": str(image_path),
                        "original_bytes_path": str(original_path),
                        "lowres_path": str(lowres_path),
                        "width": image.width,
                        "height": image.height,
                        "source_image_size": selected_row["source_image_size"],
                        "lowres_size": list(lowres.size),
                        "image_sha256": stream_digest(dataset / "train" / image_path),
                        "source_bytes_sha256": selected_row["source_bytes_sha256"],
                        "pixel_sha256": selected_row["pixel_sha256"],
                        "dhash": dhash(image),
                        "question": selected_row["question"],
                        "answers": selected_row["answers"],
                        "original_answer": selected_row["original_answer"],
                    })
                    seen.add(local_row)
        if seen != set(shard_rows):
            raise RuntimeError(f"Failed to materialize all replacements from {shard}")
    return manifests


def distribution(rows: list[dict]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(
        Counter(use_tool_flag(row) for row in rows).items(), key=lambda item: str(item[0]))}


def run(dataset: Path, source_root: Path, report_dir: Path, seed: int,
        apply: bool) -> dict:
    train_manifest = read_jsonl(dataset / "train" / "manifest.jsonl")
    annotations = read_jsonl(dataset / "train" / "annotations.jsonl")
    manifest_by_id = {row["sample_id"]: row for row in train_manifest}
    if len(manifest_by_id) != len(train_manifest):
        raise ValueError("Train manifest contains duplicate sample IDs")
    invalid_annotations = [row for row in annotations if row.get("annotation_error")]
    invalid_ids = {row["sample_id"] for row in invalid_annotations}
    if not invalid_ids:
        raise SystemExit("No invalid Train annotations found")
    if not invalid_ids <= manifest_by_id.keys():
        raise ValueError("An invalid annotation is absent from the Train manifest")
    invalid_manifest = [manifest_by_id[sample_id] for sample_id in sorted(invalid_ids)]
    targets = Counter(use_tool_flag(row) for row in invalid_manifest)

    all_manifest = read_jsonl(dataset / "manifest.jsonl")
    metadata = json.loads((dataset / "source_metadata.json").read_text())
    candidates = load_train_candidates(source_root, metadata)
    selected, rejected = select_replacements(
        candidates, dict(targets), seed,
        {row["group_id"] for row in all_manifest},
        {row["pixel_sha256"] for row in all_manifest},
        [row["dhash"] for row in all_manifest],
    )
    paired = pair_replacements(invalid_manifest, selected, seed)
    selected_ids = {f"smart-train-{row['source_row_id']:05d}" for row in paired}
    if selected_ids & {row["sample_id"] for row in all_manifest}:
        raise ValueError("A replacement sample ID already exists in the dataset")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "removed_count": len(invalid_manifest),
        "added_count": len(paired),
        "removed_use_tool": distribution(invalid_manifest),
        "added_use_tool": distribution(paired),
        "selection_rejections": dict(rejected),
        "removed_sample_ids": sorted(invalid_ids),
        "added_sample_ids": sorted(selected_ids),
        "status": "planned" if not apply else "replacement_pending_annotation",
    }
    if not apply:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    report_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(report_dir / "removed_manifest.jsonl", invalid_manifest)
    write_jsonl(report_dir / "removed_annotations.jsonl", invalid_annotations)
    added_manifest = materialize(dataset, source_root, paired)
    write_jsonl(report_dir / "added_manifest.jsonl", added_manifest)

    remaining_manifest = [row for row in train_manifest if row["sample_id"] not in invalid_ids]
    new_train_manifest = sorted(remaining_manifest + added_manifest,
                                key=lambda row: row["sample_id"])
    remaining_annotations = [row for row in annotations if row["sample_id"] not in invalid_ids]
    if len(new_train_manifest) != len(train_manifest):
        raise ValueError("Train size changed during replacement")
    if distribution(new_train_manifest) != distribution(train_manifest):
        raise ValueError("Train use_tool distribution changed during replacement")
    write_jsonl(dataset / "train" / "manifest.jsonl", new_train_manifest)
    write_jsonl(dataset / "train" / "annotations.jsonl", remaining_annotations)
    root_manifest = (new_train_manifest + read_jsonl(dataset / "val" / "manifest.jsonl")
                     + read_jsonl(dataset / "test" / "manifest.jsonl"))
    write_jsonl(dataset / "manifest.jsonl", root_manifest)

    for row in invalid_manifest:
        for key in ("image_path", "original_bytes_path", "lowres_path"):
            (dataset / "train" / row[key]).unlink(missing_ok=True)
        (dataset / "train" / "api_cache" / f"{row['sample_id']}.json").unlink(missing_ok=True)

    audit_path = dataset / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit.setdefault("replacement_history", []).append(summary)
    audit["annotation_status"] = "replacement_pending_annotation"
    write_json(audit_path, audit)
    config_path = dataset / "split_config.json"
    config = json.loads(config_path.read_text())
    config.setdefault("replacement_history", []).append(summary)
    write_json(config_path, config)
    write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(args.dataset.resolve(), args.source_root.resolve(), args.report_dir.resolve(),
        args.seed, args.apply)
