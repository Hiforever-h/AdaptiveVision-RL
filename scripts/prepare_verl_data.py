"""Convert the frozen JSONL splits into lightweight verl-agent parquet inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/visionthink_3000_300_500_balanced"
DEFAULT_OUTPUT = ROOT / "data/verl_agent"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_rows(dataset_root: Path, split: str) -> list[dict]:
    annotation_path = dataset_root / split / "annotations.jsonl"
    source = read_jsonl(annotation_path)
    seen: set[str] = set()
    rows: list[dict] = []
    for index, item in enumerate(source):
        sample_id = str(item["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in {annotation_path}: {sample_id}")
        seen.add(sample_id)
        for key in ("image_path", "lowres_path"):
            path = dataset_root / split / item[key]
            if not path.is_file():
                raise FileNotFoundError(path)

        region = item.get("region_annotation") or {}
        quality = item.get("quality_checks") or {}
        rows.append(
            {
                "data_source": "adaptive_vision",
                # The actual prompt and images are supplied by the environment. This
                # seed prompt exists only to satisfy RLHFDataset's input contract.
                "prompt": [{"role": "user", "content": "Adaptive visual acquisition."}],
                "ability": "agent",
                "extra_info": {"split": split, "index": index, "sample_id": sample_id},
                "env_kwargs": {
                    "sample_id": sample_id,
                    "split": split,
                    "question": str(item["question"]),
                    "answers": [str(answer) for answer in item["answers"]],
                    "image_path": f"{split}/{item['image_path']}",
                    "lowres_path": f"{split}/{item['lowres_path']}",
                    "reference_boxes": region.get("reference_boxes") or [],
                    "tool_reward_eligible": bool(quality.get("region_reward_eligible", False)),
                    "source_use_tool": bool((item.get("source_info") or {}).get("use_tool", False)),
                },
            }
        )
    return rows


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for split in args.splits:
        rows = build_rows(dataset_root, split)
        output = output_dir / f"{split}.parquet"
        write_parquet(output, rows)
        print(f"{split}: {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
