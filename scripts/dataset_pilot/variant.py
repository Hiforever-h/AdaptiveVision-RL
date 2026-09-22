"""Freeze the same sample into a separate run, without copying teacher outputs."""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .common import digest, read_jsonl, write_json


def prepare_variant(source, output, prompt, change_description):
    if output.exists():
        raise SystemExit("Variant output exists. Reuse its manifest; do not overwrite a run.")
    rows = read_jsonl(source / "manifest.jsonl")
    for row in rows:
        if digest((source / row["image_path"]).read_bytes()) != row["image_sha256"]:
            raise ValueError("Source image differs from frozen manifest")
    output.mkdir(parents=True)
    for name in ["manifest.jsonl", "sampling.json"]:
        shutil.copy2(source / name, output / name)
    for name in ["images", "originals", "lowres"]:
        shutil.copytree(source / name, output / name)
    shutil.copy2(prompt, output / "prompt.txt")
    baseline_files = [source / "manifest.jsonl", source / "annotations.jsonl"]
    baseline_files += sorted((source / "api_cache").glob("*.json"))
    write_json(output / "experiment.json", {
        "name": output.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_directory": str(source), "count": len(rows),
        "prompt_sha256": digest(prompt.read_bytes()),
        "baseline_sha256": {str(p.relative_to(source)): digest(p.read_bytes()) for p in baseline_files},
        "controls": "Same frozen manifest, images, and questions. No gold answers or baseline outputs in teacher requests.",
        "controlled_change": change_description,
    })
    print(f"Prepared {len(rows)} identical samples in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--change-description", default="Prompt changed; model and decoding settings unchanged.")
    args = parser.parse_args()
    prepare_variant(args.source.resolve(), args.output.resolve(), args.prompt.resolve(),
                    args.change_description)
