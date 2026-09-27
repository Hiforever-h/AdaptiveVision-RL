#!/usr/bin/env python3
"""Probe raw Qwen3-VL-Thinking completion format on verl-agent Val rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_vision_rl.prompts import INITIAL_PROMPT, SECOND_PROMPT
from adaptive_vision_rl.protocol import ParsedAction, parse_action
from adaptive_vision_rl.thinking_template import (
    ASSISTANT_PREFIX,
    configure_thinking_tokenizer,
)
from scripts.dataset_pilot.common import pixel_box
from scripts.evaluate_dtpo import (
    execute_crop,
    load_rgb,
    prepare_image,
    render_prompt,
    write_json,
    write_jsonl,
)


DEFAULT_DATASET = ROOT / "data/visionthink_3000_300_500_balanced"
DEFAULT_VAL_PARQUET = ROOT / "data/verl_agent/val.parquet"
DEFAULT_OUTPUT = Path("/root/autodl-tmp/outputs/thinking_format_probe")
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Thinking"


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    question: str
    lowres_path: Path
    image_path: Path
    reference_box: tuple[float, float, float, float] | None
    source_use_tool: bool


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str | None
    token_count: int
    first_token_id: int | None = None


def load_val_samples(
    parquet_path: Path,
    dataset_root: Path,
    *,
    offset: int,
    limit: int | None,
) -> list[ProbeSample]:
    """Use env_kwargs; the parquet prompt is only a training placeholder."""

    parquet_path = parquet_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    rows = pq.read_table(parquet_path, columns=["env_kwargs"]).to_pylist()
    selected = rows[offset : None if limit is None else offset + limit]
    if not selected:
        raise ValueError("The selected Val parquet slice is empty")

    samples: list[ProbeSample] = []
    seen: set[str] = set()
    for row in selected:
        item = row["env_kwargs"]
        if item["split"] != "val":
            raise ValueError(f"Expected Val row, found split={item['split']!r}")
        sample_id = str(item["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate Val sample_id: {sample_id}")
        seen.add(sample_id)
        lowres_path = dataset_root / str(item["lowres_path"])
        image_path = dataset_root / str(item["image_path"])
        for path in (lowres_path, image_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        boxes = item.get("reference_boxes") or []
        box = None
        if boxes:
            candidate = tuple(float(value) for value in boxes[0])
            if (
                len(candidate) == 4
                and 0 <= candidate[0] < candidate[2] <= 1
                and 0 <= candidate[1] < candidate[3] <= 1
            ):
                box = candidate
        samples.append(
            ProbeSample(
                sample_id=sample_id,
                question=str(item["question"]),
                lowres_path=lowres_path,
                image_path=image_path,
                reference_box=box,
                source_use_tool=bool(item.get("source_use_tool", False)),
            )
        )
    return samples


class BaseThinkingGenerator:
    """Load the base model only; no LoRA or training worker is required."""

    def __init__(self, args: argparse.Namespace):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code, use_fast=True
        )
        configure_thinking_tokenizer(self.processor.tokenizer)
        self.think_token_id = self.processor.tokenizer.convert_tokens_to_ids("<think>")
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
            enforce_eager=True,
            enable_chunked_prefill=False,
        )
        self.sampling = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_response_tokens,
            seed=args.seed,
        )

    def generate(
        self, texts: Sequence[str], images: Sequence[Sequence[Image.Image]]
    ) -> list[Generation]:
        if len(texts) != len(images):
            raise ValueError("Prompt and image counts differ")
        requests = [
            {
                "prompt": render_prompt(self.processor, text, len(group)),
                "multi_modal_data": {"image": list(group)},
            }
            for text, group in zip(texts, images, strict=True)
        ]
        outputs = self.llm.generate(requests, sampling_params=self.sampling, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned a different number of outputs")
        return [
            Generation(
                text=output.outputs[0].text,
                finish_reason=output.outputs[0].finish_reason,
                token_count=len(output.outputs[0].token_ids or []),
                first_token_id=(
                    output.outputs[0].token_ids[0]
                    if output.outputs[0].token_ids
                    else None
                ),
            )
            for output in outputs
        ]


def inspect_generation(
    generation: Generation, *, allow_tool: bool, image_size: tuple[int, int]
) -> tuple[dict[str, Any], ParsedAction]:
    action = parse_action(generation.text, allow_tool=allow_tool, image_size=image_size)
    result = {
        "raw_completion": generation.text,
        "finish_reason": generation.finish_reason,
        "generated_tokens": generation.token_count,
        "first_token_id": generation.first_token_id,
        "starts_with_think": generation.text.startswith("<think>"),
        "has_closing_think": "</think>" in generation.text,
        "valid_action": action.valid,
        "action_kind": action.kind,
        "parse_error": action.error,
    }
    return result, action


def _run_second_turns(
    generator,
    tasks: list[tuple[dict, list[Image.Image]]],
    *,
    batch_size: int,
    field: str,
) -> None:
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        responses = generator.generate(
            [SECOND_PROMPT.format(question=record["question"]) for record, _ in batch],
            [images for _, images in batch],
        )
        for (record, images), response in zip(batch, responses, strict=True):
            details, action = inspect_generation(
                response, allow_tool=False, image_size=images[0].size
            )
            details["valid_answer"] = action.valid and action.kind == "answer"
            record[field] = details


def run_probe(
    generator,
    samples: Sequence[ProbeSample],
    *,
    batch_size: int,
    reference_second_probes: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    samples_by_id = {sample.sample_id: sample for sample in samples}
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        low_raw = [load_rgb(sample.lowres_path) for sample in batch]
        low_images = [prepare_image(image) for image in low_raw]
        prompts = [
            INITIAL_PROMPT.format(
                question=sample.question, width=image.width, height=image.height
            )
            for sample, image in zip(batch, low_raw, strict=True)
        ]
        responses = generator.generate(prompts, [[image] for image in low_images])
        natural_second: list[tuple[dict, list[Image.Image]]] = []
        for sample, low_image, raw_image, response in zip(
            batch, low_images, low_raw, responses, strict=True
        ):
            details, action = inspect_generation(
                response, allow_tool=True, image_size=raw_image.size
            )
            record: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "source_use_tool": sample.source_use_tool,
                "first_turn": details,
                "natural_second_turn": None,
                "reference_second_turn": None,
            }
            records.append(record)
            if action.valid and action.kind == "tool" and action.bbox is not None:
                full = load_rgb(sample.image_path)
                crop, executed_box = execute_crop(full, raw_image.size, action.bbox)
                record["executed_crop_normalized"] = executed_box
                natural_second.append((record, [low_image, prepare_image(crop)]))
        _run_second_turns(
            generator, natural_second, batch_size=batch_size, field="natural_second_turn"
        )
        print(f"First turn: {len(records)}/{len(samples)}", flush=True)

    reference_tasks: list[tuple[dict, list[Image.Image]]] = []
    if reference_second_probes:
        for record in records:
            if len(reference_tasks) >= reference_second_probes:
                break
            sample = samples_by_id[record["sample_id"]]
            if record["natural_second_turn"] is not None or sample.reference_box is None:
                continue
            low_image = prepare_image(load_rgb(sample.lowres_path))
            full = load_rgb(sample.image_path)
            crop_coordinates = pixel_box(list(sample.reference_box), full.width, full.height)
            crop = full.crop(tuple(crop_coordinates))
            record["reference_crop_normalized"] = list(sample.reference_box)
            reference_tasks.append((record, [low_image, prepare_image(crop)]))
        _run_second_turns(
            generator, reference_tasks, batch_size=batch_size, field="reference_second_turn"
        )
    return records


def turn_summary(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(turns)
    if not count:
        return {"count": 0}
    return {
        "count": count,
        "starts_with_think_rate": sum(turn["starts_with_think"] for turn in turns) / count,
        "has_closing_think_rate": sum(turn["has_closing_think"] for turn in turns) / count,
        "valid_action_rate": sum(turn["valid_action"] for turn in turns) / count,
        "length_truncation_rate": sum(turn["finish_reason"] == "length" for turn in turns) / count,
        "mean_generated_tokens": sum(turn["generated_tokens"] for turn in turns) / count,
        "action_kinds": dict(Counter(turn["action_kind"] for turn in turns)),
        "finish_reasons": dict(Counter(str(turn["finish_reason"]) for turn in turns)),
        "parse_errors": dict(Counter(turn["parse_error"] for turn in turns if turn["parse_error"])),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "first_turn": turn_summary([record["first_turn"] for record in records]),
        "natural_second_turn": turn_summary(
            [record["natural_second_turn"] for record in records if record["natural_second_turn"]]
        ),
        "reference_second_turn": turn_summary(
            [
                record["reference_second_turn"]
                for record in records
                if record["reference_second_turn"]
            ]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-parquet", type=Path, default=DEFAULT_VAL_PARQUET)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-second-probes", type=int, default=8)
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=7168)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.offset < 0 or args.limit <= 0 or args.batch_size <= 0:
        raise ValueError("--offset must be nonnegative; --limit and --batch-size must be positive")
    if args.reference_second_probes < 0:
        raise ValueError("--reference-second-probes must be nonnegative")
    if args.max_response_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("Token limits must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    if args.temperature < 0:
        raise ValueError("--temperature must be nonnegative")
    if sys.platform == "darwin":
        raise RuntimeError("vLLM probing requires a Linux CUDA host")

    output_dir = args.output_dir.expanduser().resolve()
    records_path = output_dir / "completions.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and (records_path.exists() or summary_path.exists()):
        raise FileExistsError(f"Probe output exists in {output_dir}; use --overwrite")
    samples = load_val_samples(
        args.val_parquet, args.dataset_root, offset=args.offset, limit=args.limit
    )
    generator = BaseThinkingGenerator(args)
    started = time.perf_counter()
    records = run_probe(
        generator,
        samples,
        batch_size=args.batch_size,
        reference_second_probes=args.reference_second_probes,
    )
    payload = {
        "model": args.model,
        "val_parquet": str(args.val_parquet.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "assistant_prompt_suffix": ASSISTANT_PREFIX,
        "think_token_id": generator.think_token_id,
        "sampling": {
            "temperature": args.temperature,
            "max_response_tokens": args.max_response_tokens,
            "seed": args.seed,
        },
        "sample_count": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": summarize(records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records_path, records)
    write_json(summary_path, payload)
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"Raw completions: {records_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
