#!/usr/bin/env python3
"""Compare half/full adapters on Val300 by final-answer accuracy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation(*, adapter: Path, split: str, output_dir: Path,
                   dataset_root: Path, model: str, overwrite: bool,
                   batch_size: int) -> dict[str, Any]:
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_dtpo.py"),
        "--checkpoint", str(adapter),
        "--model", model,
        "--dataset-root", str(dataset_root),
        "--split", split,
        "--output-dir", str(output_dir),
        "--batch-size", str(batch_size),
    ]
    if overwrite:
        command.append("--overwrite")
    print(f"Evaluating {adapter} on {split}", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    summary = read_json(output_dir / "summary.json")
    expected = 300 if split == "val" else 500
    if summary["sample_count"] != expected or summary["split"] != split:
        raise RuntimeError(f"expected {expected} {split} samples in {output_dir}")
    if Path(summary["adapter_path"]).resolve() != adapter.resolve():
        raise RuntimeError(f"evaluation used an unexpected adapter in {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "sft/config.json")
    parser.add_argument("--model", help="Hub model ID or complete local model directory")
    parser.add_argument("--data-dir", type=Path, help="directory containing visionthink_3000_300_500_balanced/")
    parser.add_argument("--dataset-root", type=Path, help="override the Val/Test dataset directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test", action="store_true", help="test only the adapter selected on Val300")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    root = (args.checkpoint_dir or Path(config["output_dir"])).expanduser().resolve()
    output = (args.output_dir or root / "evaluation").expanduser().resolve()
    data_dir = (args.data_dir or Path(config.get("data_dir", "data"))).expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    dataset_root = (args.dataset_root or data_dir / "visionthink_3000_300_500_balanced").expanduser().resolve()
    model = args.model or config["model"]
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    effective = config["micro_batch_size"] * config["gradient_accumulation_steps"]
    half_step = round(2250 / (2 * effective))
    adapters = {
        "half": root / f"checkpoint-{half_step}",
        "final": root / "final",
    }
    for name, adapter in adapters.items():
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"{name} adapter missing: {adapter}")
    output.mkdir(parents=True, exist_ok=True)
    val_results = {
        name: run_evaluation(
            adapter=adapter, split="val", output_dir=output / f"val_{name}",
            dataset_root=dataset_root, model=model,
            overwrite=args.overwrite, batch_size=args.batch_size,
        )
        for name, adapter in adapters.items()
    }
    accuracy = {
        name: float(result["metrics"]["overall"]["accuracy"])
        for name, result in val_results.items()
    }
    # Prefer the earlier adapter on an exact tie.
    winner = "final" if accuracy["final"] > accuracy["half"] else "half"
    selection: dict[str, Any] = {
        "criterion": "Val300 final-answer accuracy only",
        "val_accuracy": accuracy,
        "selected": winner,
        "selected_adapter": str(adapters[winner]),
    }
    if args.test:
        result = run_evaluation(
            adapter=adapters[winner], split="test", output_dir=output / "test_selected",
            dataset_root=dataset_root, model=model,
            overwrite=args.overwrite, batch_size=args.batch_size,
        )
        selection["test_accuracy"] = float(result["metrics"]["overall"]["accuracy"])
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
