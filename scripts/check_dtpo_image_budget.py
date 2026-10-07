#!/usr/bin/env python3
"""Check the DTPO image prompt cap with the real Qwen3-VL processor on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "third_party/verl-agent"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adaptive_vision_rl.images import prepare_image
from adaptive_vision_rl.prompts import INITIAL_PROMPT, SECOND_PROMPT
from adaptive_vision_rl.thinking_template import (
    apply_thinking_chat_template,
    configure_thinking_tokenizer,
)
from adaptive_vision_rl.verl.image_budget import fit_image_prompt, pad_thin_crop


def longest_dataset_prompts(processor, data_root: Path, parquet_files: list[Path]):
    import pyarrow.parquet as parquet

    longest: dict[int, tuple[int, str, str]] = {}
    row_count = 0
    for parquet_file in parquet_files:
        rows = parquet.read_table(parquet_file, columns=["env_kwargs"])
        for row in rows.column("env_kwargs").to_pylist():
            row_count += 1
            with Image.open(data_root / row["lowres_path"]) as low_image:
                width, height = low_image.size
            question = str(row["question"])
            texts = (
                INITIAL_PROMPT.format(question=question, width=width, height=height),
                SECOND_PROMPT.format(question=question),
            )
            for turn, prompt_text in enumerate(texts, start=1):
                prompt = apply_thinking_chat_template(
                    processor.tokenizer, [{"role": "user", "content": prompt_text}]
                )
                length = len(processor.tokenizer.encode(prompt, add_special_tokens=False))
                if turn not in longest or length > longest[turn][0]:
                    longest[turn] = (length, prompt_text, str(row["sample_id"]))
    print(f"Scanned {row_count} dataset rows for longest text prompts")
    return longest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/root/autodl-tmp/models/qwen3vl_4b_sft_final_merged"
    )
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--scan-data", action="store_true")
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data/visionthink_3000_300_500_balanced"
    )
    parser.add_argument(
        "--parquet", type=Path, nargs="+", default=[ROOT / "data/verl_agent/train.parquet", ROOT / "data/verl_agent/val.parquet"]
    )
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    configure_thinking_tokenizer(processor.tokenizer)
    image = np.zeros((2048, 2048, 3), dtype=np.uint8)
    thin_crop = np.asarray(pad_thin_crop(Image.new("RGB", (1, 2048))))
    scenarios = [
        (INITIAL_PROMPT.format(question="Read the small text.", width=2048, height=2048), [image]),
        (SECOND_PROMPT.format(question="Read the small text."), [image, image]),
        (SECOND_PROMPT.format(question="Read the small text."), [image, thin_crop]),
    ]
    if args.scan_data:
        for turn, (_, prompt_text, sample_id) in longest_dataset_prompts(
            processor, args.data_root, args.parquet
        ).items():
            print(f"Longest dataset prompt: turn={turn} sample_id={sample_id}")
            scenarios.append((prompt_text, [image] * turn))
    for scenario, (text, images) in enumerate(scenarios, start=1):
        prompt = apply_thinking_chat_template(
            processor.tokenizer, [{"role": "user", "content": text}]
        )
        fitted = fit_image_prompt(
            prompt=prompt,
            images=images,
            tokenizer=processor.tokenizer,
            processor=processor,
            max_prompt_length=args.max_prompt_length,
            process_image=prepare_image,
        )
        encoded = processor.tokenizer(
            fitted.expanded_prompt, add_special_tokens=False
        )["input_ids"]
        if len(encoded) != fitted.prompt_length:
            raise AssertionError("tokenizer.encode and tokenizer(...) lengths differ")
        print(
            f"scenario={scenario} tokens={fitted.initial_prompt_length}->{fitted.prompt_length} "
            f"image_sizes={[item.size for item in fitted.images]}"
        )
    print("DTPO image prompt budget check passed")


if __name__ == "__main__":
    main()
