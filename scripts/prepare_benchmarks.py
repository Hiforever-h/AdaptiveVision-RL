"""Download and prepare paired original/half-size images for labeled VQA benchmarks.

Run from the repository root with the vison-rl Python environment:

    python scripts/prepare_benchmarks.py download
    python scripts/prepare_benchmarks.py extract
    python scripts/prepare_benchmarks.py verify

The source revisions are pinned so a resumed run cannot silently mix versions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1] / "data"

# DocVQA's test and MathVista's full test omit public answers, so they are
# intentionally excluded under the user's labeled-data requirement.
# MathVerse publishes testmini as its only image-bearing evaluation split.
SOURCES = {
    "ChartQA": ("HuggingFaceM4/ChartQA", "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5", "test", "data/test-*.parquet"),
    "OCRBench": ("echo840/OCRBench", "92a54bd1384387c178d5a07140a2d85e0a3d12e1", "test", "data/test-*.parquet"),
    "MME": ("lmms-lab/MME", "d6c9023f017b564f7b3ccccf5348166bce8fdbcd", "test", "data/test-*.parquet"),
    "MMVet": ("whyu/mm-vet", "8ce15108c0b55c67200eb96601ba2bb98183ff65", "test", "data/test-*.parquet"),
    "RealWorldQA": ("xai-org/RealworldQA", "17e7f75e092e47169732462ea3cdfebe911105dd", "test", "data/test-*.parquet"),
    "POPE": ("lmms-lab/POPE", "4db1276663dfa5eb8ad16a52d24c31a09e470896", "test", "data/test-*.parquet"),
    "MathVerse": ("AI4Math/MathVerse", "3bc86196678bad115a923d2851c6821dbe235939", "testmini", "testmini.parquet"),
}


def selected_names(value: str) -> list[str]:
    if value == "all":
        return list(SOURCES)
    names = [s.strip() for s in value.split(",") if s.strip()]
    unknown = set(names) - SOURCES.keys()
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return names


def get_session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "AdaptiveVision-RL-benchmark-preparation/1.0"
    return session


def fetch_file(session, url: str, target: Path, expected_size: int) -> None:
    if target.exists() and target.stat().st_size == expected_size:
        print(f"  cached {target.name} ({expected_size:,} bytes)", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    partial.unlink()
                    offset = 0
                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            actual = partial.stat().st_size
            if actual != expected_size:
                raise IOError(f"size mismatch: {actual:,} != {expected_size:,}")
            partial.replace(target)
            print(f"  downloaded {target.name} ({actual:,} bytes)", flush=True)
            return
        except Exception as exc:
            print(f"  retry {attempt + 1}/5 {target.name}: {exc}", file=sys.stderr, flush=True)
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 12))


def download(names: list[str]) -> None:
    import fnmatch

    session = get_session()
    for name in names:
        repo, revision, split, pattern = SOURCES[name]
        info_url = f"https://huggingface.co/api/datasets/{repo}"
        info_response = session.get(info_url, timeout=30)
        info_response.raise_for_status()
        current_revision = info_response.json()["sha"]
        if revision is not None and current_revision != revision:
            print(f"{name}: repository advanced from {revision} to {current_revision}; using pinned revision", flush=True)
        revision = revision or current_revision
        tree_response = session.get(
            f"https://huggingface.co/api/datasets/{repo}/tree/{revision}",
            params={"recursive": "true"},
            timeout=60,
        )
        tree_response.raise_for_status()
        files = sorted(
            (item for item in tree_response.json() if fnmatch.fnmatch(item.get("path", ""), pattern)),
            key=lambda item: item["path"],
        )
        if not files:
            raise RuntimeError(f"{name}: no files match {pattern}")
        directory = ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        provenance = {
            "dataset": name,
            "source": f"https://huggingface.co/datasets/{repo}",
            "revision": revision,
            "split": split,
            "files": [{"path": item["path"], "size": item["size"]} for item in files],
        }
        (directory / "source.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
        print(f"{name}: {repo}@{revision[:12]} {split}, {len(files)} file(s)", flush=True)
        for item in files:
            relative = item["path"]
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(relative, safe='/')}"
            fetch_file(session, url, directory / "source" / relative, item["size"])


def image_payload(row: dict) -> tuple[bytes, str | None, str]:
    """Return the single embedded image; fail instead of dropping extra images."""
    found: list[tuple[bytes, str | None, str]] = []

    def visit(value, key: str) -> None:
        if isinstance(value, dict):
            raw = value.get("bytes")
            if isinstance(raw, bytes):
                found.append((raw, value.get("path"), key))
                return
            for subkey, subvalue in value.items():
                visit(subvalue, f"{key}.{subkey}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{key}[{index}]")

    for key, value in row.items():
        visit(value, key)
    if len(found) != 1:
        raise ValueError(f"expected exactly one embedded image, found {len(found)} in {list(row)}")
    return found[0]


def strip_image_bytes(value):
    if isinstance(value, dict):
        return {key: strip_image_bytes(item) for key, item in value.items() if key != "bytes"}
    if isinstance(value, list):
        return [strip_image_bytes(item) for item in value]
    if isinstance(value, bytes):
        return {"binary_length": len(value)}
    return value


def save_images(raw: bytes, directory: Path) -> tuple[str, str, list[int], list[int], str]:
    from PIL import Image, ImageOps

    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as opened:
        image_format = opened.format
        if image_format == "JPEG":
            extension = ".jpg"
        elif image_format == "PNG":
            extension = ".png"
        elif image_format == "WEBP":
            extension = ".webp"
        elif image_format == "GIF":
            extension = ".gif"
        else:
            extension = "." + (image_format or "bin").lower()
        high_rel = f"images/high/{digest[:2]}/{digest}{extension}"
        low_extension = extension if image_format in {"JPEG", "PNG", "WEBP"} else ".png"
        low_rel = f"images/low/{digest[:2]}/{digest}{low_extension}"
        high_path, low_path = directory / high_rel, directory / low_rel
        high_path.parent.mkdir(parents=True, exist_ok=True)
        low_path.parent.mkdir(parents=True, exist_ok=True)
        if not high_path.exists():
            temp = high_path.with_name(high_path.name + ".part")
            temp.write_bytes(raw)
            temp.replace(high_path)
        oriented = ImageOps.exif_transpose(opened)
        high_size = list(oriented.size)
        low_size = [max(1, width // 2) for width in high_size]
        if not low_path.exists():
            smaller = oriented.resize(tuple(low_size), Image.Resampling.LANCZOS)
            temp = low_path.with_name(low_path.name + ".part")
            if low_extension == ".jpg":
                smaller.convert("RGB").save(temp, format="JPEG", quality=95, subsampling=0)
            elif low_extension == ".webp":
                smaller.save(temp, format="WEBP", quality=95)
            else:
                smaller.save(temp, format="PNG", optimize=True)
            temp.replace(low_path)
    return high_rel, low_rel, high_size, low_size, digest


def extract(names: list[str]) -> None:
    import pyarrow.parquet as parquet
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    for name in names:
        directory = ROOT / name
        provenance = json.loads((directory / "source.json").read_text())
        paths = [directory / "source" / item["path"] for item in provenance["files"]]
        if any(not path.exists() for path in paths):
            raise FileNotFoundError(f"{name}: source parquet missing")
        expected = sum(parquet.ParquetFile(path).metadata.num_rows for path in paths)
        output = directory / "samples.jsonl"
        if output.exists() and sum(1 for _ in output.open()) == expected:
            print(f"{name}: {expected:,} existing records; use verify to check them", flush=True)
            continue
        print(f"{name}: extracting {expected:,} questions", flush=True)
        temp = output.with_name(output.name + ".part")
        row_index = 0
        with temp.open("w", encoding="utf-8") as writer:
            for path in paths:
                reader = parquet.ParquetFile(path)
                for batch in reader.iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        raw, original_path, image_field = image_payload(row)
                        high, low, high_size, low_size, digest = save_images(raw, directory)
                        metadata = strip_image_bytes(row)
                        record = {
                            "sample_id": f"{name}-{row_index:06d}",
                            "dataset": name,
                            "split": provenance["split"],
                            "question": next((row[key] for key in ("question", "query", "text", "query_wo") if row.get(key)), None),
                            "answer": next((row[key] for key in ("answer", "answers", "label") if row.get(key) is not None), None),
                            "source_file": str(path.relative_to(directory)),
                            "image_field": image_field,
                            "source_image_path": original_path,
                            "high_image": high,
                            "low_image": low,
                            "high_size": high_size,
                            "low_size": low_size,
                            "image_sha256": digest,
                            "data": metadata,
                        }
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                        row_index += 1
                print(f"  {path.name}: {row_index:,}/{expected:,}", flush=True)
        if row_index != expected:
            raise RuntimeError(f"{name}: wrote {row_index}, expected {expected}")
        temp.replace(output)
        print(f"{name}: extracted {row_index:,} records", flush=True)


def verify(names: list[str]) -> None:
    import pyarrow.parquet as parquet
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = None
    for name in names:
        directory = ROOT / name
        provenance = json.loads((directory / "source.json").read_text())
        expected = sum(
            parquet.ParquetFile(directory / "source" / item["path"]).metadata.num_rows
            for item in provenance["files"]
        )
        seen: set[str] = set()
        count = 0
        with (directory / "samples.jsonl").open() as manifest:
            for line in manifest:
                record = json.loads(line)
                count += 1
                if record["sample_id"] in seen:
                    raise ValueError(f"{name}: duplicate sample ID {record['sample_id']}")
                seen.add(record["sample_id"])
                if not record.get("question"):
                    raise ValueError(f"{name}: missing question for {record['sample_id']}")
                if record.get("answer") in (None, "", [], [""]):
                    raise ValueError(f"{name}: missing public answer for {record['sample_id']}")
                high_path = directory / record["high_image"]
                low_path = directory / record["low_image"]
                if not high_path.is_file() or not low_path.is_file():
                    raise FileNotFoundError(f"{name}: missing image for {record['sample_id']}")
                with Image.open(high_path) as high, Image.open(low_path) as low:
                    display_size = ImageOps.exif_transpose(high).size
                    expected_low = tuple(max(1, side // 2) for side in display_size)
                    if tuple(record["high_size"]) != display_size:
                        raise ValueError(f"{name}: high size mismatch for {record['sample_id']}")
                    if low.size != expected_low or tuple(record["low_size"]) != expected_low:
                        raise ValueError(f"{name}: low size mismatch for {record['sample_id']}")
        if count != expected:
            raise ValueError(f"{name}: {count} records, expected {expected}")
        print(f"{name}: verified {count:,} question/image pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "extract", "verify"])
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names, or all")
    args = parser.parse_args()
    names = selected_names(args.datasets)
    if args.action == "download":
        download(names)
    elif args.action == "extract":
        extract(names)
    else:
        verify(names)


if __name__ == "__main__":
    main()
