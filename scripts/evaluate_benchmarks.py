#!/usr/bin/env python3
"""Evaluate the frozen six-benchmark set with direct or one-crop inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_vision_rl.prompts import INITIAL_PROMPT, SECOND_PROMPT
from adaptive_vision_rl.protocol import parse_action
from adaptive_vision_rl.thinking_template import apply_thinking_chat_template
from adaptive_vision_rl.verl.image_budget import fit_image_prompt
from scripts.dataset_pilot.common import answer_check
from scripts.evaluate_dtpo import (
    VLLMEvaluator,
    execute_crop,
    file_sha256,
    load_rgb,
    prepare_image,
    resolve_lora_adapter,
    write_json,
    write_jsonl,
)

DATASET_NAMES = ("ChartQA", "OCRBench", "MME", "RealWorldQA", "POPE", "MathVerse")
DEFAULT_DATASET_ROOT = ROOT / "data/eval_6bench_3000_v1"
HIGH_ONLY_PROMPT = """<image>
You see a high-resolution full image and a question.

Question: {question}

Your response MUST start with <think> and contain a non-empty </think> block.
Briefly reason from the visible image. After </think>, output exactly
<answer>short answer</answer>.

You cannot call a tool. Do not omit <think>, repeat either block, or write
anything outside the required tags.
"""


@dataclass(frozen=True)
class BenchmarkSample:
    eval_id: str
    dataset: str
    question: str
    answers: list[str]
    high_path: Path
    low_path: Path
    question_type: str | None


def load_benchmark_samples(root: Path, limit: int | None = None) -> tuple[list[BenchmarkSample], Path]:
    root = root.expanduser().resolve()
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    listed = selection.get("datasets", {})
    if set(listed) != set(DATASET_NAMES):
        raise ValueError(f"selection.json must contain exactly {DATASET_NAMES}")
    samples: list[BenchmarkSample] = []
    seen: set[str] = set()
    for name in DATASET_NAMES:
        directory = root / name
        manifest = directory / "samples.jsonl"
        if file_sha256(manifest) != listed[name]["subset_manifest_sha256"]:
            raise ValueError(f"manifest checksum mismatch: {manifest}")
        count = 0
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                eval_id = str(row["eval_id"])
                if row["dataset"] != name or eval_id in seen:
                    raise ValueError(f"wrong dataset or duplicate eval_id: {eval_id}")
                seen.add(eval_id)
                answers = [str(value) for value in row["answers"]]
                if not row["question"].strip() or not answers or not all(a.strip() for a in answers):
                    raise ValueError(f"missing question or answer: {eval_id}")
                high = (directory / row["high_image"]).resolve()
                low = (directory / row["low_image"]).resolve()
                if not high.is_relative_to(directory) or not low.is_relative_to(directory):
                    raise ValueError(f"image path escapes dataset directory: {eval_id}")
                if not high.is_file() or not low.is_file():
                    raise FileNotFoundError(f"missing image pair for {eval_id}")
                samples.append(BenchmarkSample(
                    eval_id=eval_id,
                    dataset=name,
                    question=str(row["question"]),
                    answers=answers,
                    high_path=high,
                    low_path=low,
                    question_type=(row.get("source_data") or {}).get("question_type"),
                ))
                count += 1
        if count != listed[name]["questions"]:
            raise ValueError(f"question count mismatch for {name}: {count}")
    if len(samples) != selection["total_questions"]:
        raise ValueError("selection total_questions does not match the manifests")
    if limit is not None:
        # Round robin keeps a small smoke run representative of all datasets.
        groups = [[s for s in samples if s.dataset == name] for name in DATASET_NAMES]
        samples = [sample for row in zip(*groups) for sample in row][:limit]
    return samples, selection_path


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = " ".join(text.split())
    return text.rstrip(".。")


def _number(value: str, *, chart: bool = False) -> Decimal | None:
    text = _normalize(value).replace(",", "")
    if chart:
        text = text.strip("$% ")
    text = text.rstrip("°")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def score_answer(sample: BenchmarkSample, prediction: str | None) -> tuple[bool, str]:
    if prediction is None:
        return False, "invalid_format"
    references = sample.answers
    checked = answer_check(prediction, references)
    if checked["match"]:
        return True, checked["method"]
    pred = _normalize(prediction)
    if sample.dataset == "ChartQA":
        number = _number(prediction, chart=True)
        for reference in references:
            target = _number(reference, chart=True)
            if number is not None and target is not None:
                if abs(number - target) <= abs(target) * Decimal("0.05"):
                    return True, "chartqa_numeric_5pct"
    if sample.dataset == "OCRBench":
        # OCRBench's public evaluator accepts reference text appearing in the
        # prediction; the prompt still asks the model for a short answer.
        compact_pred = "".join(pred.split())
        for reference in references:
            compact_ref = "".join(_normalize(reference).split())
            if compact_ref and compact_ref in compact_pred:
                return True, "ocrbench_substring"
    if sample.dataset == "MathVerse":
        if sample.question_type == "multi-choice":
            for reference in references:
                letter = _normalize(reference).upper()
                if letter in {"A", "B", "C", "D", "E"}:
                    if re.fullmatch(r"(?:option|choice|answer(?: is)?)\s*[:：]?\s*" + letter, pred, re.I):
                        return True, "mathverse_choice_letter"
        number = _number(prediction)
        if number is not None and any(number == _number(ref) for ref in references):
            return True, "mathverse_numeric_exact"
    return False, "unmatched"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_block(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    direct = [r for r in records if not r["used_tool"]]
    tool = [r for r in records if r["used_tool"]]
    return {
        "count": len(records),
        "correct_count": sum(bool(r["correct"]) for r in records),
        "accuracy": _mean([float(r["correct"]) for r in records]),
        "direct_answer_accuracy": _mean([float(r["correct"]) for r in direct]),
        "tool_answer_accuracy": _mean([float(r["correct"]) for r in tool]),
        "tool_call_rate": _mean([float(r["used_tool"]) for r in records]),
        "format_compliance": _mean([float(r["format_compliance"]) for r in records]),
        "invalid_first_action_rate": _mean([float(not r["first_action_valid"]) for r in records]),
        "invalid_final_answer_rate": _mean([float(not r["final_answer_valid"]) for r in records]),
        "vision_tokens_low_first": _mean([float(r["vision_tokens_low_first"]) for r in records]),
        "vision_tokens_low_second": _mean([float(r["vision_tokens_low_second"]) for r in records]),
        "vision_tokens_crop": _mean([float(r["vision_tokens_crop"]) for r in records]),
        "vision_tokens_full": _mean([float(r["vision_tokens_full"]) for r in records]),
        "vision_tokens_acquired": _mean([float(r["vision_tokens_acquired"]) for r in records]),
        "vision_tokens_processed": _mean([float(r["vision_tokens_processed"]) for r in records]),
        "vision_ratio": _mean([float(r["vision_ratio"]) for r in records]),
        "vision_ratio_true": _mean([float(r["vision_ratio_true"]) for r in records]),
    }


def summarize_mode(records: Sequence[dict[str, Any]], generation_seconds: float, elapsed: float) -> dict[str, Any]:
    by_dataset = {
        name: metric_block([record for record in records if record["dataset"] == name])
        for name in DATASET_NAMES
    }
    return {
        "overall": metric_block(records),
        "by_dataset": by_dataset,
        "macro_accuracy": _mean([m["accuracy"] for m in by_dataset.values() if m["accuracy"] is not None]),
        "timing": {
            "generation_seconds": generation_seconds,
            "evaluation_seconds": elapsed,
            "generation_throughput_samples_per_second": len(records) / generation_seconds if generation_seconds else None,
            "end_to_end_throughput_samples_per_second": len(records) / elapsed if elapsed else None,
        },
    }


def _checked_generate(evaluator: Any, prompts: Sequence[str], images: Sequence[Sequence[Image.Image]]) -> tuple[list[str], list[list[int]], float]:
    started = time.perf_counter()
    responses = evaluator.generate(prompts, images)
    seconds = time.perf_counter() - started
    counts = evaluator.last_image_token_counts
    if len(responses) != len(prompts) or len(counts) != len(prompts):
        raise RuntimeError("model returned an unexpected batch length")
    for expected, group in zip(images, counts, strict=True):
        if len(expected) != len(group) or any(token <= 0 for token in group):
            raise RuntimeError("invalid image token counts returned by evaluator")
    return responses, counts, seconds


def full_image_token_count(evaluator: Any, question: str, image: Image.Image) -> int:
    """Count the actual fitted high-only image that forms the ratio baseline."""
    prompt = apply_thinking_chat_template(
        evaluator.processor.tokenizer,
        [{"role": "user", "content": HIGH_ONLY_PROMPT.format(question=question)}],
    )
    fitted = fit_image_prompt(
        prompt=prompt,
        images=[image],
        tokenizer=evaluator.processor.tokenizer,
        processor=evaluator.processor,
        max_prompt_length=evaluator.max_prompt_length,
        process_image=prepare_image,
    )
    return fitted.vision_tokens[0]


def evaluate_batch(evaluator: Any, samples: Sequence[BenchmarkSample], mode: str) -> tuple[list[dict[str, Any]], float]:
    states: list[dict[str, Any]] = []
    prompts: list[str] = []
    image_groups: list[list[Image.Image]] = []
    for sample in samples:
        full_raw = load_rgb(sample.high_path)
        low_raw = load_rgb(sample.low_path) if mode == "low_tool" else None
        states.append({"sample": sample, "full_raw": full_raw, "low": low_raw,
                       "low_size": low_raw.size if low_raw is not None else None,
                       "full_tokens": full_image_token_count(evaluator, sample.question, full_raw) if mode == "low_tool" else None})
        if mode == "low_tool":
            prompts.append(INITIAL_PROMPT.format(question=sample.question, width=low_raw.width, height=low_raw.height))
            image_groups.append([low_raw])
        elif mode == "high_only":
            prompts.append(HIGH_ONLY_PROMPT.format(question=sample.question))
            image_groups.append([full_raw])
        else:
            raise ValueError(f"unknown mode: {mode}")
    first, first_counts, first_seconds = _checked_generate(evaluator, prompts, image_groups)
    tool_states: list[dict[str, Any]] = []
    second_prompts: list[str] = []
    second_images: list[list[Image.Image]] = []
    for state, response, counts in zip(states, first, first_counts, strict=True):
        action = parse_action(response, allow_tool=mode == "low_tool", image_size=state["low_size"] or state["full_raw"].size)
        state.update(first_response=response, first_action=action, first_tokens=counts[0],
                     used_tool=action.valid and action.kind == "tool", second_tokens=0,
                     crop_tokens=0, second_response=None, second_action=None, crop_box=None)
        if not state["used_tool"]:
            continue
        crop, executed = execute_crop(state["full_raw"], state["low_size"], action.bbox)
        state["crop_box"] = executed
        tool_states.append(state)
        second_prompts.append(SECOND_PROMPT.format(question=state["sample"].question))
        second_images.append([state["low"], crop])
    second_seconds = 0.0
    if tool_states:
        second, second_counts, second_seconds = _checked_generate(evaluator, second_prompts, second_images)
        for state, response, counts in zip(tool_states, second, second_counts, strict=True):
            state["second_response"] = response
            state["second_action"] = parse_action(response, allow_tool=False, image_size=state["low_size"])
            state["second_tokens"], state["crop_tokens"] = counts
    records: list[dict[str, Any]] = []
    for state in states:
        sample = state["sample"]
        first_action = state["first_action"]
        final_action = state["second_action"] if state["used_tool"] else first_action
        final_valid = bool(final_action.valid and final_action.kind == "answer")
        prediction = final_action.answer if final_valid else None
        correct, method = score_answer(sample, prediction)
        full_tokens = state["full_tokens"] or state["first_tokens"]
        low_first = state["first_tokens"] if mode == "low_tool" else 0
        low_second = state["second_tokens"]
        crop_tokens = state["crop_tokens"]
        acquired = low_first + crop_tokens if mode == "low_tool" else full_tokens
        processed = acquired + low_second
        format_compliance = (float(first_action.valid) + float(final_valid)) / 2 if state["used_tool"] else float(final_valid)
        records.append({
            "eval_id": sample.eval_id, "dataset": sample.dataset, "mode": mode,
            "question": sample.question, "references": sample.answers, "prediction": prediction,
            "correct": correct, "answer_score": float(correct), "answer_match_method": method,
            "used_tool": state["used_tool"], "first_action_valid": first_action.valid,
            "final_answer_valid": final_valid, "first_parse_error": first_action.error,
            "final_parse_error": final_action.error, "format_compliance": format_compliance,
            "first_response": state["first_response"], "second_response": state["second_response"],
            "crop_box": state["crop_box"], "vision_tokens_low_first": low_first,
            "vision_tokens_low_second": low_second, "vision_tokens_crop": crop_tokens,
            "vision_tokens_full": full_tokens, "vision_tokens_acquired": acquired,
            "vision_tokens_processed": processed, "vision_ratio": acquired / full_tokens,
            "vision_ratio_true": processed / full_tokens,
        })
    return records, first_seconds + second_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="base model path or Hugging Face model ID")
    parser.add_argument("--adapter", type=Path, help="optional LoRA adapter directory or checkpoint root")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("low_tool", "high_only"), required=True,
                        help="run exactly one image-input mode")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="smoke run: round-robin prefix across datasets")
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=9216)
    parser.add_argument("--max-num-batched-tokens", type=int, default=10240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if sys.platform == "darwin":
        raise RuntimeError("vLLM evaluation requires a Linux CUDA host")
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--batch-size and --limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    if args.max_model_len <= args.max_response_tokens:
        raise ValueError("--max-model-len must exceed --max-response-tokens")
    output = args.output_dir.expanduser().resolve()
    paths = [output / "summary.json", output / f"predictions_{args.mode}.jsonl"]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError(f"evaluation output exists in {output}; use --overwrite")
    samples, selection_path = load_benchmark_samples(args.dataset_root, args.limit)
    adapter = resolve_lora_adapter(args.adapter) if args.adapter is not None else None
    model_args = argparse.Namespace(
        model=args.base_model, base_model=adapter is None, merged_model=False,
        trust_remote_code=args.trust_remote_code, max_model_len=args.max_model_len,
        max_response_tokens=args.max_response_tokens, dtype=args.dtype, seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    print(f"Model: {args.base_model}; adapter: {adapter or 'none'}; samples: {len(samples)}", flush=True)
    evaluator = VLLMEvaluator(model_args, adapter)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Mode: {args.mode}", flush=True)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    generation_seconds = 0.0
    for start in range(0, len(samples), args.batch_size):
        batch, seconds = evaluate_batch(evaluator, samples[start:start + args.batch_size], args.mode)
        records.extend(batch)
        generation_seconds += seconds
        if len(records) % max(args.batch_size * 10, 1) == 0 or len(records) == len(samples):
            print(f"  {len(records)}/{len(samples)} accuracy={metric_block(records)['accuracy']:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    write_jsonl(output / f"predictions_{args.mode}.jsonl", records)
    mode_metrics = summarize_mode(records, generation_seconds, elapsed)
    summary = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model, "adapter_path": str(adapter) if adapter else None,
        "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors") if adapter else None,
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "selection_sha256": file_sha256(selection_path), "sample_count": len(samples),
        "partial": len(samples) < 3000, "mode": args.mode, "batch_size": args.batch_size,
        "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                     "max_response_tokens": args.max_response_tokens, "seed": args.seed},
        "metrics": {args.mode: mode_metrics},
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({args.mode: mode_metrics["overall"]}, ensure_ascii=False, indent=2), flush=True)
    print(f"Summary: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
