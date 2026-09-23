#!/usr/bin/env python3
"""Deterministic two-turn evaluation for a trained AdaptiveVision LoRA policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_vision_rl.prompts import INITIAL_PROMPT, SECOND_PROMPT
from adaptive_vision_rl.protocol import ParsedAction, extract_answer_candidate, parse_action
from scripts.dataset_pilot.common import answer_check, pixel_box, read_jsonl
from scripts.dataset_pilot.reward import geometry_reward


DEFAULT_DATASET = ROOT / "data/visionthink_3000_300_500_balanced"
DEFAULT_CHECKPOINT = Path("/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/outputs/evaluation/qwen3vl_4b_dtpo_lora_test")
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    question: str
    answers: list[str]
    image_path: Path
    lowres_path: Path
    reference_boxes: list[list[float]]
    tool_reward_eligible: bool
    source_use_tool: bool


def _adapter_is_complete(path: Path) -> bool:
    weights = path / "adapter_model.safetensors"
    return (path / "adapter_config.json").is_file() and weights.is_file()


def _step_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("global_step_"))
    except ValueError:
        return -1


def resolve_lora_adapter(checkpoint: Path) -> Path:
    """Resolve a PEFT adapter from an adapter, actor, step, or run directory."""

    checkpoint = checkpoint.expanduser().resolve()
    direct_candidates = [
        checkpoint,
        checkpoint / "lora_adapter",
        checkpoint / "actor" / "lora_adapter",
    ]
    for candidate in direct_candidates:
        if _adapter_is_complete(candidate):
            return candidate

    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        raw_step = tracker.read_text(encoding="utf-8").strip()
        if raw_step.isdigit():
            candidate = checkpoint / f"global_step_{raw_step}" / "actor" / "lora_adapter"
            if _adapter_is_complete(candidate):
                return candidate

    if checkpoint.is_dir():
        steps = sorted(
            (path for path in checkpoint.glob("global_step_*") if path.is_dir()),
            key=_step_number,
            reverse=True,
        )
        for step in steps:
            candidate = step / "actor" / "lora_adapter"
            if _adapter_is_complete(candidate):
                return candidate

    raise FileNotFoundError(
        "Could not find adapter_config.json and adapter_model.safetensors under "
        f"{checkpoint}. Expected a LoRA adapter directory or a verl checkpoint root."
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_samples(
    dataset_root: Path,
    split: str,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[EvalSample], Path]:
    dataset_root = dataset_root.expanduser().resolve()
    annotation_path = dataset_root / split / "annotations.jsonl"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    rows = read_jsonl(annotation_path)
    selected = rows[offset : None if limit is None else offset + limit]
    samples: list[EvalSample] = []
    seen: set[str] = set()
    for row in selected:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in evaluation slice: {sample_id}")
        seen.add(sample_id)
        image_path = dataset_root / split / str(row["image_path"])
        lowres_path = dataset_root / split / str(row["lowres_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not lowres_path.is_file():
            raise FileNotFoundError(lowres_path)
        region = row.get("region_annotation") or {}
        quality = row.get("quality_checks") or {}
        boxes = region.get("reference_boxes") or []
        samples.append(
            EvalSample(
                sample_id=sample_id,
                question=str(row["question"]),
                answers=[str(value) for value in row["answers"]],
                image_path=image_path,
                lowres_path=lowres_path,
                reference_boxes=boxes,
                tool_reward_eligible=bool(
                    quality.get("region_reward_eligible", False) and boxes
                ),
                source_use_tool=bool((row.get("source_info") or {}).get("use_tool", False)),
            )
        )
    if not samples:
        raise ValueError("the requested evaluation slice is empty")
    return samples, annotation_path


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def prepare_image(
    image: Image.Image,
    *,
    max_pixels: int = 2048 * 2048,
    min_pixels: int = 256 * 256,
) -> Image.Image:
    """Match verl-agent's rollout-side image size normalization."""

    result = image.convert("RGB")
    area = result.width * result.height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        result = result.resize((int(result.width * scale), int(result.height * scale)))
    elif area < min_pixels:
        scale = math.sqrt(min_pixels / area)
        result = result.resize((int(result.width * scale), int(result.height * scale)))
    return result


def vision_token_count(processor: Any, image: Image.Image) -> int:
    inputs = processor.image_processor([image], return_tensors="pt")
    grid = inputs["image_grid_thw"]
    merge = int(processor.image_processor.merge_size) ** 2
    return int(sum(int(item.prod().item()) // merge for item in grid))


def render_prompt(processor: Any, text: str, image_count: int) -> str:
    if text.count("<image>") != image_count:
        raise ValueError(
            f"prompt/image mismatch: {text.count('<image>')} placeholders for "
            f"{image_count} images"
        )
    chat = [{"role": "user", "content": text}]
    prompt = processor.tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return prompt.replace("<image>", IMAGE_TOKEN)


def action_format_score(action: ParsedAction, expected_kind: str) -> float:
    if not action.valid or action.kind != expected_kind:
        return 0.0
    return 1.0 if action.has_think else 0.5


def execute_crop(
    full_image: Image.Image,
    low_size: tuple[int, int],
    bbox: tuple[float, float, float, float],
) -> tuple[Image.Image, list[float]]:
    low_width, low_height = low_size
    normalized = [
        bbox[0] / low_width,
        bbox[1] / low_height,
        bbox[2] / low_width,
        bbox[3] / low_height,
    ]
    crop_coordinates = pixel_box(normalized, full_image.width, full_image.height)
    crop = full_image.crop(tuple(crop_coordinates))
    executed = [
        crop_coordinates[0] / full_image.width,
        crop_coordinates[1] / full_image.height,
        crop_coordinates[2] / full_image.width,
        crop_coordinates[3] / full_image.height,
    ]
    return crop, executed


class VLLMEvaluator:
    def __init__(self, args: argparse.Namespace, adapter_path: Path):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VLLM_USE_V1", "1")
        if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(
            os.environ.get("OMP_NUM_THREADS", "0")
        ) <= 0:
            os.environ["OMP_NUM_THREADS"] = "1"

        from transformers import AutoProcessor

        from adaptive_vision_rl.vllm_compat import (
            install_qwen3vl_lora_mapping_backport,
        )

        install_qwen3vl_lora_mapping_backport()
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        adapter_config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        rank = int(adapter_config.get("r", 0))
        if rank <= 0:
            raise ValueError(f"invalid LoRA rank in {adapter_path / 'adapter_config.json'}")

        self.processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
        )
        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            limit_mm_per_prompt={"image": 2},
            enable_lora=True,
            max_loras=1,
            max_lora_rank=rank,
            enforce_eager=True,
            enable_chunked_prefill=False,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_response_tokens,
            seed=args.seed,
        )
        self.lora_request = LoRARequest("adaptive_vision_dtpo", 1, str(adapter_path))

    def generate(
        self,
        texts: Sequence[str],
        images: Sequence[Sequence[Image.Image]],
    ) -> list[str]:
        if len(texts) != len(images):
            raise ValueError("text and image batch sizes differ")
        requests = []
        for text, image_group in zip(texts, images, strict=True):
            requests.append(
                {
                    "prompt": render_prompt(self.processor, text, len(image_group)),
                    "multi_modal_data": {"image": list(image_group)},
                }
            )
        outputs = self.llm.generate(
            requests,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned a different number of outputs than requests")
        return [output.outputs[0].text for output in outputs]


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def metric_block(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tool_records = [record for record in records if record["used_tool"]]
    direct_records = [record for record in records if not record["used_tool"]]
    eligible_tool_records = [
        record
        for record in tool_records
        if record["tool_reward_eligible"] and record["tool_reward"] is not None
    ]
    return {
        "count": len(records),
        "accuracy": _mean(float(record["correct"]) for record in records),
        "direct_answer_accuracy": _mean(
            float(record["correct"]) for record in direct_records
        ),
        "tool_answer_accuracy": _mean(
            float(record["correct"]) for record in tool_records
        ),
        "tool_call_rate": _mean(float(record["used_tool"]) for record in records),
        "format_compliance": _mean(
            float(record["format_compliance"]) for record in records
        ),
        "invalid_first_action_rate": _mean(
            float(not record["first_action_valid"]) for record in records
        ),
        "invalid_final_answer_rate": _mean(
            float(not record["final_answer_valid"]) for record in records
        ),
        "outcome_reward": _mean(float(record["outcome_reward"]) for record in records),
        "tool_reward": _mean(
            float(record["tool_reward"]) for record in eligible_tool_records
        ),
        "tool_reward_count": len(eligible_tool_records),
        "vision_tokens_low": _mean(float(record["vision_tokens_low"]) for record in records),
        "vision_tokens_crop": _mean(
            float(record["vision_tokens_crop"]) for record in records
        ),
        "vision_tokens_acquired": _mean(
            float(record["vision_tokens_acquired"]) for record in records
        ),
        "vision_tokens_processed": _mean(
            float(record["vision_tokens_processed"]) for record in records
        ),
        "vision_token_ratio": _mean(
            float(record["vision_token_ratio"]) for record in records
        ),
        "estimated_generation_seconds_per_sample": _mean(
            float(record["estimated_generation_seconds"]) for record in records
        ),
    }


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    generation_seconds: float,
    evaluation_seconds: float,
) -> dict[str, Any]:
    source_false = [record for record in records if not record["source_use_tool"]]
    source_true = [record for record in records if record["source_use_tool"]]
    return {
        "overall": metric_block(records),
        "by_source_use_tool": {
            "false": metric_block(source_false),
            "true": metric_block(source_true),
        },
        "timing": {
            "generation_seconds": generation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "generation_throughput_samples_per_second": (
                len(records) / generation_seconds if generation_seconds else None
            ),
            "end_to_end_throughput_samples_per_second": (
                len(records) / evaluation_seconds if evaluation_seconds else None
            ),
        },
    }


def _finalize_record(
    state: dict[str, Any],
    second_response: str | None = None,
    second_seconds_per_sample: float = 0.0,
) -> dict[str, Any]:
    sample: EvalSample = state["sample"]
    first_action: ParsedAction = state["first_action"]
    if state["used_tool"]:
        final_action = parse_action(
            second_response or "",
            allow_tool=False,
            image_size=state["low_size"],
        )
        prediction = final_action.answer or extract_answer_candidate(second_response or "")
        format_reward = 0.5 * (
            state["first_format_score"] + action_format_score(final_action, "answer")
        ) / 2.0
        final_valid = final_action.valid and final_action.kind == "answer"
        final_error = final_action.error
    else:
        final_action = first_action
        prediction = first_action.answer or extract_answer_candidate(state["first_response"])
        format_reward = 0.5 * state["first_format_score"]
        final_valid = first_action.valid and first_action.kind == "answer"
        final_error = first_action.error

    check = (
        answer_check(prediction, sample.answers)
        if prediction is not None
        else {"match": False, "method": "missing_answer"}
    )
    acquired = state["vision_tokens_low"] + state["vision_tokens_crop"]
    processed = (
        state["vision_tokens_low"]
        if not state["used_tool"]
        else 2 * state["vision_tokens_low"] + state["vision_tokens_crop"]
    )
    full_tokens = state["vision_tokens_full"]
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "references": sample.answers,
        "prediction": prediction,
        "correct": bool(check["match"]),
        "answer_match_method": check["method"],
        "source_use_tool": sample.source_use_tool,
        "used_tool": state["used_tool"],
        "first_action_valid": first_action.valid,
        "final_answer_valid": final_valid,
        "first_parse_error": first_action.error,
        "final_parse_error": final_error,
        "first_response": state["first_response"],
        "second_response": second_response,
        "format_reward": format_reward,
        "format_compliance": format_reward / 0.5,
        "outcome_reward": float(check["match"]) + format_reward,
        "tool_reward_eligible": sample.tool_reward_eligible,
        "predicted_box": state["predicted_box"],
        "reference_boxes": sample.reference_boxes,
        "coverage": state["coverage"],
        "iou": state["iou"],
        "tool_reward": state["tool_reward"],
        "vision_tokens_low": state["vision_tokens_low"],
        "vision_tokens_crop": state["vision_tokens_crop"],
        "vision_tokens_acquired": acquired,
        "vision_tokens_processed": processed,
        "vision_tokens_full": full_tokens,
        "vision_token_ratio": acquired / full_tokens if full_tokens else 0.0,
        "estimated_generation_seconds": (
            state["first_seconds_per_sample"] + second_seconds_per_sample
        ),
    }


def evaluate_batch(
    evaluator: VLLMEvaluator,
    samples: Sequence[EvalSample],
    *,
    coverage_weight: float,
) -> tuple[list[dict[str, Any]], float]:
    states: list[dict[str, Any]] = []
    first_prompts: list[str] = []
    first_images: list[list[Image.Image]] = []
    for sample in samples:
        low_raw = load_rgb(sample.lowres_path)
        full_raw = load_rgb(sample.image_path)
        low_processed = prepare_image(low_raw)
        full_processed = prepare_image(full_raw)
        low_size = low_raw.size
        state = {
            "sample": sample,
            "low_raw": low_raw,
            "full_raw": full_raw,
            "low_processed": low_processed,
            "low_size": low_size,
            "vision_tokens_low": vision_token_count(evaluator.processor, low_processed),
            "vision_tokens_full": vision_token_count(evaluator.processor, full_processed),
            "vision_tokens_crop": 0,
            "predicted_box": None,
            "coverage": None,
            "iou": None,
            "tool_reward": None,
        }
        states.append(state)
        first_prompts.append(
            INITIAL_PROMPT.format(
                question=sample.question,
                width=low_size[0],
                height=low_size[1],
            )
        )
        first_images.append([low_processed])

    started = time.perf_counter()
    first_responses = evaluator.generate(first_prompts, first_images)
    first_seconds = time.perf_counter() - started
    first_per_sample = first_seconds / len(states)

    tool_states: list[dict[str, Any]] = []
    second_prompts: list[str] = []
    second_images: list[list[Image.Image]] = []
    for state, response in zip(states, first_responses, strict=True):
        action = parse_action(
            response,
            allow_tool=True,
            image_size=state["low_size"],
        )
        state["first_response"] = response
        state["first_action"] = action
        state["first_seconds_per_sample"] = first_per_sample
        state["used_tool"] = bool(
            action.valid and action.kind == "tool" and action.bbox is not None
        )
        expected = "tool" if state["used_tool"] else "answer"
        state["first_format_score"] = action_format_score(action, expected)
        if not state["used_tool"]:
            continue

        crop, executed = execute_crop(
            state["full_raw"], state["low_size"], action.bbox
        )
        crop_processed = prepare_image(crop)
        state["predicted_box"] = executed
        state["vision_tokens_crop"] = vision_token_count(
            evaluator.processor, crop_processed
        )
        sample: EvalSample = state["sample"]
        if sample.tool_reward_eligible:
            geometry = geometry_reward(
                executed,
                sample.reference_boxes,
                coverage_weight=coverage_weight,
            )
            if geometry is not None:
                state["coverage"] = geometry["coverage"]
                state["iou"] = geometry["iou"]
                state["tool_reward"] = geometry["reward"]
        tool_states.append(state)
        second_prompts.append(SECOND_PROMPT.format(question=sample.question))
        second_images.append([state["low_processed"], crop_processed])

    second_responses: list[str] = []
    second_seconds = 0.0
    if tool_states:
        started = time.perf_counter()
        second_responses = evaluator.generate(second_prompts, second_images)
        second_seconds = time.perf_counter() - started
    second_per_sample = second_seconds / len(tool_states) if tool_states else 0.0
    response_by_sample = {
        state["sample"].sample_id: response
        for state, response in zip(tool_states, second_responses, strict=True)
    }
    records = [
        _finalize_record(
            state,
            second_response=response_by_sample.get(state["sample"].sample_id),
            second_seconds_per_sample=(second_per_sample if state["used_tool"] else 0.0),
        )
        for state in states
    ]
    return records, first_seconds + second_seconds


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the latest DTPO LoRA checkpoint on the frozen test split."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=6656)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if sys.platform == "darwin":
        raise RuntimeError("vLLM evaluation requires a Linux CUDA host")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.offset < 0:
        raise ValueError("--offset cannot be negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    if not 0 < args.coverage_weight < 1:
        raise ValueError("--coverage-weight must be between 0 and 1")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    adapter_path = resolve_lora_adapter(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    results_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and (results_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"evaluation output already exists in {output_dir}; pass --overwrite to replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, annotation_path = load_samples(
        args.dataset_root,
        args.split,
        offset=args.offset,
        limit=args.limit,
    )
    print(f"Resolved LoRA adapter: {adapter_path}", flush=True)
    print(f"Evaluation samples: {len(samples)} from {annotation_path}", flush=True)

    evaluator = VLLMEvaluator(args, adapter_path)
    records: list[dict[str, Any]] = []
    generation_seconds = 0.0
    evaluation_started = time.perf_counter()
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        batch_records, batch_generation_seconds = evaluate_batch(
            evaluator,
            batch,
            coverage_weight=args.coverage_weight,
        )
        records.extend(batch_records)
        generation_seconds += batch_generation_seconds
        print(
            f"Evaluated {len(records)}/{len(samples)} "
            f"(accuracy={metric_block(records)['accuracy']:.4f})",
            flush=True,
        )
    evaluation_seconds = time.perf_counter() - evaluation_started

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter_path": str(adapter_path),
        "adapter_sha256": file_sha256(adapter_path / "adapter_model.safetensors"),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "annotation_path": str(annotation_path),
        "annotation_sha256": file_sha256(annotation_path),
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "sample_count": len(records),
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_response_tokens": args.max_response_tokens,
            "seed": args.seed,
        },
        "metrics": summarize(
            records,
            generation_seconds=generation_seconds,
            evaluation_seconds=evaluation_seconds,
        ),
    }
    write_jsonl(results_path, records)
    write_json(summary_path, payload)
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"Predictions: {results_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
