"""Download only a small, deterministic sample from the HF dataset viewer.

The viewer currently exposes a partial conversion. This is a diagnostic pilot,
NOT a full-dataset random sample or a finalized train/dev/test split.
"""
import argparse
import io
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import DEFAULT_OUTPUT, digest, write_json, write_jsonl

DATASET = "Senqiao/VisionThink-Smart-Train"
WINDOWS = [0, 5500, 8500, 9500, 11000, 13000, 16000, 18400]


def dhash(image) -> int:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(small.getdata())
    return sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


def prepare(output: Path, count: int, seed: int):
    if (output / "manifest.jsonl").exists():
        raise SystemExit("Manifest exists; reuse it or choose another --output. No resampling.")
    if count < 2 or count > 50:
        raise SystemExit("Pilot size must be 2..50; bulk preparation needs a full-source audit.")
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])))
    revision_url = f"https://huggingface.co/api/datasets/{DATASET}"
    response = session.get(revision_url, timeout=45)
    response.raise_for_status()
    revision = response.json()["sha"]
    candidates = []
    windows_metadata = []
    for offset in WINDOWS:
        response = session.get("https://datasets-server.huggingface.co/rows", params={
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset, "length": 100}, timeout=60)
        response.raise_for_status()
        data = response.json()
        write_json(output / "source_cache" / f"rows_{offset}.json", data)
        windows_metadata.append({"offset": offset, "rows": len(data["rows"]),
                                 "partial": data.get("partial"),
                                 "viewer_rows_total": data["num_rows_total"],
                                 "viewer_revision": response.headers.get("x-revision")})
        for item in data["rows"]:
            row = item["row"]
            if row.get("problem", "").strip() and row.get("solution", "").strip() and row.get("images"):
                candidates.append({"row_index": item["row_idx"], "window": offset, **row})
        print(f"candidate window {offset}: {len(data['rows'])} rows", flush=True)
    # Round-robin over windows within each tool stratum, randomizing within windows.
    rng = random.Random(seed)
    queues = {}
    for use_tool in [False, True]:
        buckets = []
        for offset in WINDOWS:
            bucket = [r for r in candidates if r["window"] == offset and r["info"]["use_tool"] == use_tool]
            rng.shuffle(bucket)
            buckets.append(bucket)
        queues[use_tool] = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    queues[use_tool].append(bucket.pop())
    rows, rejected = [], []
    seen_hashes, seen_docs, seen_dhashes = set(), set(), []
    for use_tool, quota in [(False, count // 2), (True, count - count // 2)]:
        accepted = 0
        for row in queues[use_tool]:
            if accepted >= quota:
                break
            source_id = str(row["doc_id"])
            if source_id in seen_docs:
                rejected.append({"row": row["row_index"], "reason": "repeated_doc_id"})
                continue
            # Public dataset image only; no API credentials are used for downloads.
            response = session.get(row["images"]["src"], timeout=60)
            response.raise_for_status()
            blob = response.content
            image = Image.open(io.BytesIO(blob))
            image.load()
            if image.size != (row["images"]["width"], row["images"]["height"]):
                raise ValueError("Viewer image dimensions do not match metadata")
            source_size = image.size
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            pixel_hash = digest(f"{width}x{height}:".encode() + image.tobytes())
            near_hash = dhash(image)
            if pixel_hash in seen_hashes or any((near_hash ^ h).bit_count() <= 4 for h in seen_dhashes):
                rejected.append({"row": row["row_index"], "reason": "exact_or_near_duplicate_image"})
                continue
            sample_id = f"smart-train-{row['row_index']:05d}"
            image_path = Path("images") / f"{sample_id}.png"
            original_path = Path("originals") / f"{sample_id}.bin"
            for directory in ["images", "originals", "lowres"]:
                (output / directory).mkdir(exist_ok=True)
            (output / original_path).write_bytes(blob)
            image.save(output / image_path)
            low = image.copy()
            low.thumbnail((row["info"]["tgt_width"], row["info"]["tgt_height"]), Image.Resampling.LANCZOS)
            low_path = Path("lowres") / f"{sample_id}.png"
            low.save(output / low_path)
            rows.append({
                "sample_id": sample_id, "source": DATASET, "source_revision_observed": revision,
                "source_access": "HF viewer partial conversion; original asset bytes retained",
                "source_row_id": row["row_index"], "source_window": row["window"],
                "doc_id": source_id, "group_id": f"pilot-doc:{source_id}",
                "split": "pilot_train_reserved", "formal_split_audit_pending": True,
                "data_source": row["data_source"], "source_info": row["info"],
                "image_path": str(image_path), "original_bytes_path": str(original_path),
                "lowres_path": str(low_path), "width": width, "height": height,
                "source_image_size": list(source_size), "lowres_size": list(low.size),
                "image_sha256": digest((output / image_path).read_bytes()),
                "source_bytes_sha256": digest(blob), "pixel_sha256": pixel_hash,
                "dhash": f"{near_hash:016x}",
                "question": row["problem"].replace("<image>", "").strip(),
                "answers": [row["solution"]], "original_answer": row["original_answer"],
            })
            seen_docs.add(source_id)
            seen_hashes.add(pixel_hash)
            seen_dhashes.append(near_hash)
            accepted += 1
            print(f"prepared {len(rows)}/{count}: {sample_id}, tool_hint={use_tool}, {width}x{height}", flush=True)
        if accepted < quota:
            raise RuntimeError(f"Only {accepted}/{quota} valid rows for use_tool={use_tool}")
    response = session.get(revision_url, timeout=45)
    response.raise_for_status()
    if response.json()["sha"] != revision:
        raise RuntimeError("Source revision changed while preparing; manifest not frozen")
    write_jsonl(output / "manifest.jsonl", rows)
    write_json(output / "sampling.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "count": count,
        "dataset": DATASET, "revision_observed_before_and_after": revision,
        "method": "balanced use_tool strata, round-robin windows, seeded shuffle, image/doc dedup",
        "windows": windows_metadata, "candidate_count": len(candidates), "rejected": rejected,
        "tool_hint_counts": dict(Counter(str(r["source_info"]["use_tool"]) for r in rows)),
        "limitation": "Partial viewer sample; not population representative. Viewer cannot be pinned by revision; cached rows, bytes and hashes are the reproducibility record.",
        "future_splits": {"train": {"source": DATASET, "target": 3000},
                          "dev": {"source": DATASET, "target": 300},
                          "test": {"source": "Senqiao/VisionThink-Smart-Val", "target": 500}},
        "formal_split_status": "not_created; full-source doc/image/near-duplicate audit required",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.count, args.seed)
