#!/usr/bin/env python3
"""Compare half/full adapters on Val300 by final-answer accuracy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation(*, adapter: Path, split: str, output_dir: Path,
                   dataset_root: Path, model: str, overwrite: bool,
                   batch_size: int, merged_model: Path | None = None) -> dict[str, Any]:
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_dtpo.py"),
        "--checkpoint", str(adapter),
        "--model", str(merged_model) if merged_model is not None else model,
        "--dataset-root", str(dataset_root),
        "--split", split,
        "--output-dir", str(output_dir),
        "--batch-size", str(batch_size),
    ]
    if overwrite:
        command.append("--overwrite")
    if merged_model is not None:
        command.append("--merged-model")
    print(f"Evaluating {adapter} on {split}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluator.log"
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            exit_code = process.wait()
    if exit_code:
        raise RuntimeError(
            f"evaluation failed with exit code {exit_code}; the complete vLLM log is at {log_path}"
        )
    summary = read_json(output_dir / "summary.json")
    expected = 300 if split == "val" else 500
    if summary["sample_count"] != expected or summary["split"] != split:
        raise RuntimeError(f"expected {expected} {split} samples in {output_dir}")
    if Path(summary["adapter_path"]).resolve() != adapter.resolve():
        raise RuntimeError(f"evaluation used an unexpected adapter in {output_dir}")
    return summary


def evaluate_adapter(*, adapter: Path, split: str, output_dir: Path,
                     dataset_root: Path, model: str, overwrite: bool,
                     batch_size: int, dynamic_lora: bool,
                     temporary_root: Path) -> dict[str, Any]:
    common = dict(
        adapter=adapter, split=split, output_dir=output_dir,
        dataset_root=dataset_root, model=model,
        overwrite=overwrite, batch_size=batch_size,
    )
    if dynamic_lora:
        return run_evaluation(**common)
    with tempfile.TemporaryDirectory(prefix="sft-merged-", dir=temporary_root) as temp:
        merged = Path(temp) / "model"
        command = [
            sys.executable, "-m", "sft.merge_lora",
            "--base-model", model,
            "--adapter", str(adapter),
            "--output-dir", str(merged),
        ]
        print(f"Temporarily merging {adapter} for vLLM evaluation", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        return run_evaluation(**common, merged_model=merged)


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
    parser.add_argument("--dynamic-lora", action="store_true",
                        help="use vLLM dynamic LoRA instead of temporary merged models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    root = (args.checkpoint_dir or Path(config["output_dir"])).expanduser().resolve()
    if (root / "adapter_config.json").is_file():
        raise ValueError(
            f"--checkpoint-dir must be the training output directory containing "
            f"checkpoint-70/ and final/, not the adapter directory {root}; "
            f"use {root.parent}"
        )
    output = (args.output_dir or root / "evaluation").expanduser().resolve()
    data_dir = (args.data_dir or Path(config.get("data_dir", "data"))).expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    dataset_root = (args.dataset_root or data_dir / "visionthink_3000_300_500_balanced").expanduser().resolve()
    model = args.model or config["model"]
    if Path(model).is_absolute():
        model_dir = Path(model).expanduser()
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"--model must be a complete local model directory containing "
                f"config.json: {model_dir}. To use the Hugging Face cache, pass "
                f"--model Qwen/Qwen3-VL-4B-Thinking instead."
            )
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
        name: evaluate_adapter(
            adapter=adapter, split="val", output_dir=output / f"val_{name}",
            dataset_root=dataset_root, model=model,
            overwrite=args.overwrite, batch_size=args.batch_size,
            dynamic_lora=args.dynamic_lora, temporary_root=output,
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
        "evaluation_backend": "dynamic_lora" if args.dynamic_lora else "temporary_merged_model",
    }
    if args.test:
        result = evaluate_adapter(
            adapter=adapters[winner], split="test", output_dir=output / "test_selected",
            dataset_root=dataset_root, model=model,
            overwrite=args.overwrite, batch_size=args.batch_size,
            dynamic_lora=args.dynamic_lora, temporary_root=output,
        )
        selection["test_accuracy"] = float(result["metrics"]["overall"]["accuracy"])
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
