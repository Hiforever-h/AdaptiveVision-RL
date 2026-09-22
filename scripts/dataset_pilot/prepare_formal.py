"""Build frozen 3000/300/500 splits from the pinned full Arrow sources.

The script downloads resumable source shards, scans every row, groups by doc_id,
rejects exact/near duplicate images across all selected splits, and materializes
only the selected rows as original bytes, normalized PNGs, and low-res views.
Train can optionally be stratified by the source ``use_tool`` hint. It does not
call an annotation model.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import ROOT, read_jsonl, write_json, write_jsonl

SOURCES = {
    "smart_train": "Senqiao/VisionThink-Smart-Train",
    "smart_val": "Senqiao/VisionThink-Smart-Val",
}
DEFAULT_OUTPUT = ROOT / "data/visionthink_3000_300_500"


def session():
    value = requests.Session()
    value.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"])))
    return value


def stream_digest(path, chunk_size=8 * 1024 * 1024):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_score(seed, *parts):
    text = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode()).hexdigest()


def image_hash(image):
    hasher = hashlib.sha256()
    hasher.update(f"{image.width}x{image.height}:".encode())
    hasher.update(image.tobytes())
    return hasher.hexdigest()


def dhash(image):
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(small.getdata())
    value = sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    return f"{value:016x}"


def repo_metadata(repo):
    response = session().get(f"https://huggingface.co/api/datasets/{repo}", timeout=60)
    response.raise_for_status()
    data = response.json()
    files = sorted(item["rfilename"] for item in data["siblings"]
                   if item["rfilename"].endswith(".arrow"))
    if not files:
        raise RuntimeError(f"No Arrow files found for {repo}")
    return {"repo": repo, "revision": data["sha"], "files": files}


def download_shard(repo, revision, filename, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    client = session()
    head = client.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    expected = int(head.headers["content-length"])
    if destination.exists():
        if destination.stat().st_size != expected:
            raise RuntimeError(f"Existing shard has wrong size: {destination}")
        return {"filename": filename, "bytes": expected,
                "sha256": stream_digest(destination), "downloaded": False}
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = client.get(url, headers=headers, stream=True, allow_redirects=True,
                          timeout=(20, 300))
    if offset and response.status_code == 416 and offset == expected:
        partial.replace(destination)
    else:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        completed = offset
        next_notice = completed + 256 * 1024 * 1024
        with partial.open(mode) as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    completed += len(chunk)
                    if completed >= next_notice:
                        print(f"downloading {filename}: {completed / 2**30:.2f}/{expected / 2**30:.2f} GiB",
                              flush=True)
                        next_notice += 256 * 1024 * 1024
        if partial.stat().st_size != expected:
            raise RuntimeError(f"Incomplete shard {filename}: {partial.stat().st_size}/{expected}")
        partial.replace(destination)
    return {"filename": filename, "bytes": expected,
            "sha256": stream_digest(destination), "downloaded": True}


def download_sources(output, metadata, workers):
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for source_key, source in metadata.items():
            directory = output / "source_cache" / source_key
            for filename in source["files"]:
                future = pool.submit(download_shard, source["repo"], source["revision"],
                                     filename, directory / filename)
                tasks.append((source_key, future))
        records = defaultdict(list)
        for source_key, future in tasks:
            record = future.result()
            records[source_key].append(record)
            print(f"source ready {source_key}/{record['filename']}: "
                  f"{record['bytes'] / 2**20:.1f} MiB", flush=True)
    return {key: sorted(value, key=lambda item: item["filename"])
            for key, value in records.items()}


def scan_shard(path, source_key, repo, revision, scan_dir):
    candidate_path = scan_dir / source_key / f"{path.name}.jsonl"
    meta_path = scan_dir / source_key / f"{path.name}.meta.json"
    if candidate_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    candidates, rejected = [], Counter()
    row_count = 0
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            columns = {name: batch.column(index) for index, name in enumerate(batch.schema.names)}
            for index in range(batch.num_rows):
                row_in_shard = row_count
                row_count += 1
                row = {name: column[index].as_py() for name, column in columns.items()}
                problem = (row.get("problem") or "").replace("<image>", "").strip()
                solution = (row.get("solution") or "").strip()
                image_value = row.get("images") or {}
                blob = image_value.get("bytes")
                if not problem or not solution or not blob:
                    rejected["missing_image_question_or_solution"] += 1
                    continue
                try:
                    original = Image.open(io.BytesIO(blob))
                    original.load()
                    source_size = list(original.size)
                    normalized = ImageOps.exif_transpose(original).convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError):
                    rejected["invalid_image"] += 1
                    continue
                info = row.get("info") or {}
                if not info.get("tgt_width") or not info.get("tgt_height"):
                    rejected["missing_target_size"] += 1
                    continue
                doc_id = str(row.get("doc_id") or "").strip()
                pixels = image_hash(normalized)
                candidates.append({
                    "source_key": source_key, "source": repo, "source_revision": revision,
                    "source_shard": path.name, "source_row_in_shard": row_in_shard,
                    "doc_id": doc_id, "group_key": f"doc:{doc_id}" if doc_id else f"image:{pixels}",
                    "question": problem, "answers": [solution],
                    "original_answer": row.get("original_answer"),
                    "data_source": row.get("data_source"), "source_info": info,
                    "source_image_size": source_size,
                    "width": normalized.width, "height": normalized.height,
                    "source_bytes_sha256": hashlib.sha256(blob).hexdigest(),
                    "pixel_sha256": pixels, "dhash": dhash(normalized),
                })
    write_jsonl(candidate_path, candidates)
    meta = {"source_key": source_key, "source_shard": path.name,
            "row_count": row_count, "candidate_count": len(candidates),
            "rejected": dict(rejected)}
    write_json(meta_path, meta)
    print(f"scanned {source_key}/{path.name}: {len(candidates)}/{row_count} candidates", flush=True)
    return meta


def scan_sources(output, metadata):
    scan_dir = output / "scan_index"
    shard_meta = defaultdict(list)
    for source_key, source in metadata.items():
        for filename in source["files"]:
            path = output / "source_cache" / source_key / filename
            result = scan_shard(path, source_key, source["repo"], source["revision"], scan_dir)
            shard_meta[source_key].append(result)
    candidates = defaultdict(list)
    for source_key, source in metadata.items():
        offset = 0
        for item in sorted(shard_meta[source_key], key=lambda value: value["source_shard"]):
            path = scan_dir / source_key / f"{item['source_shard']}.jsonl"
            for candidate in read_jsonl(path):
                candidate["source_row_id"] = offset + candidate["source_row_in_shard"]
                candidates[source_key].append(candidate)
            offset += item["row_count"]
        source["row_count_scanned"] = offset
    return candidates, shard_meta


def representative_groups(candidates, seed, source_key):
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["group_key"]].append(candidate)
    representatives = []
    for group_key, rows in groups.items():
        representatives.append(min(rows, key=lambda row: stable_score(
            seed, source_key, group_key, row["source_row_id"])))
    return representatives, groups


def near_duplicate(value, selected, threshold=4):
    integer = int(value, 16)
    return any((integer ^ int(other, 16)).bit_count() <= threshold for other in selected)


def choose(rows, count, seed, label, occupied_groups, occupied_pixels, occupied_dhashes,
           blocked_groups=None):
    blocked_groups = blocked_groups or set()
    selected, rejected = [], Counter()
    ordered = sorted(rows, key=lambda row: stable_score(seed, label, row["group_key"]))
    for row in ordered:
        if len(selected) >= count:
            break
        if row["group_key"] in blocked_groups or row["group_key"] in occupied_groups:
            rejected["group_overlap"] += 1
            continue
        if row["pixel_sha256"] in occupied_pixels:
            rejected["exact_duplicate_image"] += 1
            continue
        if near_duplicate(row["dhash"], occupied_dhashes):
            rejected["near_duplicate_image"] += 1
            continue
        chosen = {**row, "split": label,
                  "selection_score": stable_score(seed, label, row["group_key"])}
        selected.append(chosen)
        occupied_groups.add(row["group_key"])
        occupied_pixels.add(row["pixel_sha256"])
        occupied_dhashes.append(row["dhash"])
    if len(selected) != count:
        raise RuntimeError(f"Only selected {len(selected)}/{count} rows for {label}")
    return selected, rejected


def use_tool_flag(row):
    value = row["source_info"].get("use_tool")
    return value is True or value == 1 or str(value).lower() == "true"


def choose_use_tool_stratified(rows, targets, seed, label, occupied_groups,
                               occupied_pixels, occupied_dhashes, blocked_groups=None):
    """Choose one deterministic stream while enforcing exact use_tool quotas."""
    blocked_groups = blocked_groups or set()
    targets = {bool(key): int(value) for key, value in targets.items()}
    selected, rejected, selected_counts = [], Counter(), Counter()
    ordered = sorted(rows, key=lambda row: stable_score(seed, label, row["group_key"]))
    for row in ordered:
        stratum = use_tool_flag(row)
        if stratum not in targets or selected_counts[stratum] >= targets[stratum]:
            continue
        if row["group_key"] in blocked_groups or row["group_key"] in occupied_groups:
            rejected["group_overlap"] += 1
            continue
        if row["pixel_sha256"] in occupied_pixels:
            rejected["exact_duplicate_image"] += 1
            continue
        if near_duplicate(row["dhash"], occupied_dhashes):
            rejected["near_duplicate_image"] += 1
            continue
        chosen = {**row, "split": label,
                  "selection_score": stable_score(seed, label, row["group_key"])}
        selected.append(chosen)
        selected_counts[stratum] += 1
        occupied_groups.add(row["group_key"])
        occupied_pixels.add(row["pixel_sha256"])
        occupied_dhashes.append(row["dhash"])
        if all(selected_counts[key] == value for key, value in targets.items()):
            break
    if any(selected_counts[key] != value for key, value in targets.items()):
        actual = {str(key): selected_counts[key] for key in targets}
        expected = {str(key): value for key, value in targets.items()}
        raise RuntimeError(f"Only selected use_tool strata {actual}; expected {expected}")
    return selected, rejected


def pilot_groups(path):
    if not path.exists():
        return set()
    return {f"doc:{row['doc_id']}" for row in read_jsonl(path) if row.get("doc_id")}


def select_splits(candidates, seed, train_count, val_count, test_count, excluded_groups,
                  train_require_high_count=None):
    test_reps, test_groups = representative_groups(candidates["smart_val"], seed, "smart_val")
    train_reps, train_groups = representative_groups(candidates["smart_train"], seed, "smart_train")
    occupied_groups, occupied_pixels, occupied_dhashes = set(), set(), []
    test, test_rejected = choose(test_reps, test_count, seed, "test", occupied_groups,
                                 occupied_pixels, occupied_dhashes)
    val, val_rejected = choose(train_reps, val_count, seed, "val", occupied_groups,
                               occupied_pixels, occupied_dhashes, excluded_groups)
    if train_require_high_count is None:
        train, train_rejected = choose(
            train_reps, train_count, seed, "train", occupied_groups,
            occupied_pixels, occupied_dhashes, excluded_groups)
        requested_train_distribution = None
    else:
        train, train_rejected = choose_use_tool_stratified(
            train_reps,
            {False: train_count - train_require_high_count,
             True: train_require_high_count},
            seed, "train", occupied_groups, occupied_pixels, occupied_dhashes,
            excluded_groups)
        requested_train_distribution = {
            "False": train_count - train_require_high_count,
            "True": train_require_high_count,
        }
    stats = {
        "candidate_rows": {key: len(value) for key, value in candidates.items()},
        "document_groups": {"smart_train": len(train_groups), "smart_val": len(test_groups)},
        "pilot_groups_excluded_from_train_and_val": len(excluded_groups),
        "requested_train_use_tool": requested_train_distribution,
        "selection_rejections": {"test": dict(test_rejected), "val": dict(val_rejected),
                                 "train": dict(train_rejected)},
    }
    return {"train": train, "val": val, "test": test}, stats


def materialize(output, splits, metadata, source_root=None):
    source_root = source_root or output
    selected_by_location = defaultdict(dict)
    for rows in splits.values():
        for row in rows:
            selected_by_location[(row["source_key"], row["source_shard"])][
                row["source_row_in_shard"]] = row
    manifests = {name: [] for name in splits}
    for (source_key, shard), wanted in sorted(selected_by_location.items()):
        path = source_root / "source_cache" / source_key / shard
        seen = set()
        row_number = 0
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                images = batch.column(batch.schema.get_field_index("images"))
                for index in range(batch.num_rows):
                    local_row = row_number
                    row_number += 1
                    if local_row not in wanted:
                        continue
                    selected = wanted[local_row]
                    blob = images[index].as_py()["bytes"]
                    if hashlib.sha256(blob).hexdigest() != selected["source_bytes_sha256"]:
                        raise ValueError("Source bytes changed between scan and materialization")
                    image = Image.open(io.BytesIO(blob))
                    image.load()
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    if image_hash(image) != selected["pixel_sha256"]:
                        raise ValueError("Pixels changed between scan and materialization")
                    split = selected["split"]
                    source_label = "train" if source_key == "smart_train" else "val"
                    sample_id = f"smart-{source_label}-{selected['source_row_id']:05d}"
                    split_dir = output / split
                    image_path = Path("images") / f"{sample_id}.png"
                    original_path = Path("originals") / f"{sample_id}.bin"
                    lowres_path = Path("lowres") / f"{sample_id}.png"
                    for directory in [image_path.parent, original_path.parent, lowres_path.parent]:
                        (split_dir / directory).mkdir(parents=True, exist_ok=True)
                    (split_dir / original_path).write_bytes(blob)
                    image.save(split_dir / image_path)
                    lowres = image.copy()
                    lowres.thumbnail((selected["source_info"]["tgt_width"],
                                      selected["source_info"]["tgt_height"]),
                                     Image.Resampling.LANCZOS)
                    lowres.save(split_dir / lowres_path)
                    manifest = {
                        "sample_id": sample_id, "source": selected["source"],
                        "source_revision": selected["source_revision"],
                        "source_access": "pinned full Arrow shard; original asset bytes retained",
                        "source_shard": shard,
                        "source_row_in_shard": selected["source_row_in_shard"],
                        "source_row_id": selected["source_row_id"],
                        "doc_id": selected["doc_id"], "group_id": selected["group_key"],
                        "split": split, "formal_split_audit_pending": False,
                        "selection_seed": selected.get("selection_seed"),
                        "selection_score": selected["selection_score"],
                        "data_source": selected["data_source"],
                        "source_info": selected["source_info"],
                        "image_path": str(image_path),
                        "original_bytes_path": str(original_path),
                        "lowres_path": str(lowres_path),
                        "width": image.width, "height": image.height,
                        "source_image_size": selected["source_image_size"],
                        "lowres_size": list(lowres.size),
                        "image_sha256": stream_digest(split_dir / image_path),
                        "source_bytes_sha256": selected["source_bytes_sha256"],
                        "pixel_sha256": selected["pixel_sha256"], "dhash": selected["dhash"],
                        "question": selected["question"], "answers": selected["answers"],
                        "original_answer": selected["original_answer"],
                    }
                    manifests[split].append(manifest)
                    seen.add(local_row)
        if seen != set(wanted):
            raise RuntimeError(f"Failed to materialize rows from {source_key}/{shard}")
        print(f"materialized {source_key}/{shard}: {len(seen)} selected rows", flush=True)
    for split, rows in manifests.items():
        rows.sort(key=lambda row: row["sample_id"])
        write_jsonl(output / split / "manifest.jsonl", rows)
    all_rows = [row for split in ["train", "val", "test"] for row in manifests[split]]
    write_jsonl(output / "manifest.jsonl", all_rows)
    return manifests


def distribution(rows, key):
    if key == "use_tool":
        values = (str(row["source_info"]["use_tool"]) for row in rows)
    else:
        values = (str(row.get(key)) for row in rows)
    return dict(sorted(Counter(values).items()))


def audit(manifests, selection_stats, shard_meta, source_files, metadata, seed):
    all_rows = [row for rows in manifests.values() for row in rows]
    if len({row["group_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Document group overlap remains across selected splits")
    if len({row["pixel_sha256"] for row in all_rows}) != len(all_rows):
        raise ValueError("Exact image duplicate remains across selected splits")
    dhashes = [int(row["dhash"], 16) for row in all_rows]
    for index, value in enumerate(dhashes):
        if any((value ^ other).bit_count() <= 4 for other in dhashes[:index]):
            raise ValueError("Near duplicate image remains across selected splits")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "sources": metadata, "source_files": source_files,
        "source_scan": {key: {
            "rows": sum(item["row_count"] for item in values),
            "candidates": sum(item["candidate_count"] for item in values),
            "rejected": dict(sum((Counter(item["rejected"]) for item in values), Counter())),
        } for key, values in shard_meta.items()},
        **selection_stats,
        "splits": {split: {"count": len(rows),
                            "use_tool": distribution(rows, "use_tool"),
                            "data_source": distribution(rows, "data_source")}
                   for split, rows in manifests.items()},
        "cross_split_checks": {"group_overlap": 0, "exact_pixel_duplicates": 0,
                               "near_duplicates_dhash_hamming_le_4": 0},
        "annotation_status": "not_started",
        "teacher_default_for_future_annotation": {
            "provider": "zai", "model": "glm-5.3-flash",
            "thinking": "enabled", "reasoning_effort": "max",
            "coordinate_format": "pixels",
        },
    }


def prepare(output, seed, train_count, val_count, test_count, workers, exclude_manifest,
            train_require_high_count=None, local_source_root=None):
    if (output / "manifest.jsonl").exists():
        raise SystemExit("Formal manifest exists; refusing to resample or overwrite it")
    output.mkdir(parents=True, exist_ok=True)
    if local_source_root:
        metadata = json.loads((local_source_root / "source_metadata.json").read_text())
        source_files = json.loads((local_source_root / "audit.json").read_text())["source_files"]
        source_root = local_source_root
    else:
        metadata = {key: repo_metadata(repo) for key, repo in SOURCES.items()}
        source_root = output
    write_json(output / "source_metadata.json", metadata)
    if not local_source_root:
        source_files = download_sources(output, metadata, workers)
    candidates, shard_meta = scan_sources(source_root, metadata)
    excluded = pilot_groups(exclude_manifest)
    splits, selection_stats = select_splits(candidates, seed, train_count, val_count,
                                             test_count, excluded,
                                             train_require_high_count)
    for rows in splits.values():
        for row in rows:
            row["selection_seed"] = seed
    manifests = materialize(output, splits, metadata, source_root)
    if not local_source_root:
        for key, source in metadata.items():
            current = repo_metadata(source["repo"])["revision"]
            if current != source["revision"]:
                raise RuntimeError(f"Source revision changed during preparation: {source['repo']}")
    report = audit(manifests, selection_stats, shard_meta, source_files, metadata, seed)
    write_json(output / "audit.json", report)
    write_json(output / "split_config.json", {
        "seed": seed, "train": train_count, "val": val_count, "test": test_count,
        "method": "full-source scan; one deterministic row per doc_id; test from Smart-Val; "
                  "val/train from Smart-Train; exact and dHash<=4 image dedup across all splits; "
                  "optional exact Train use_tool stratification",
        "train_require_high_count": train_require_high_count,
        "excluded_manifest": str(exclude_manifest),
        "local_source_root": str(local_source_root) if local_source_root else None,
        "gold_or_teacher_performance_used_for_selection": False,
        "annotation_performed": False,
    })
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--train-count", type=int, default=3000)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--train-require-high-count", type=int,
                        help="Exact Train count whose source use_tool hint is true")
    parser.add_argument("--download-workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--local-source-root", type=Path,
                        help="Reuse pinned source_cache and scan_index from an existing build")
    parser.add_argument("--exclude-manifest", type=Path,
                        default=ROOT / "data/pilot20/manifest.jsonl")
    args = parser.parse_args()
    if min(args.train_count, args.val_count, args.test_count) < 1:
        parser.error("All split counts must be positive")
    if (args.train_require_high_count is not None and
            not 0 <= args.train_require_high_count <= args.train_count):
        parser.error("--train-require-high-count must be between 0 and --train-count")
    prepare(args.output.resolve(), args.seed, args.train_count, args.val_count,
            args.test_count, args.download_workers, args.exclude_manifest.resolve(),
            args.train_require_high_count,
            args.local_source_root.resolve() if args.local_source_root else None)
