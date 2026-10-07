"""Qwen3-VL turn dataset and multimodal collator for the 2,250-turn SFT run."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from adaptive_vision_rl.images import prepare_image
from adaptive_vision_rl.thinking_template import apply_thinking_chat_template


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TURNS = ROOT / "data/sft_adaptive_vision_v1/train_2250_turns/turns.jsonl"


def resolve_data_path(path: str | Path, data_dir: Path) -> Path:
    """Map JSONL paths beginning with data/ into a movable data directory."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == "data":
        candidate = Path(*candidate.parts[1:])
    return (data_dir / candidate).resolve()


def load_turns(path: Path, *, data_dir: Path = ROOT / "data",
               validate_images: bool = True) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    counts: Counter[tuple[str, str]] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            route = row["route"]
            turn_index = row["turn_index"]
            stage = row["stage"]
            if route not in {"direct", "tool"} or turn_index not in {0, 1}:
                raise ValueError(f"bad route or turn at {path}:{line_number}")
            expected = ("direct", 0, "decision") if route == "direct" else (
                ("tool", 0, "decision") if turn_index == 0 else ("tool", 1, "answer_after_tool")
            )
            if (route, turn_index, stage) != expected:
                raise ValueError(f"bad stage at {path}:{line_number}")
            key = (sample_id, turn_index)
            if key in keys:
                raise ValueError(f"duplicate turn {key} at {path}:{line_number}")
            keys.add(key)
            images = row["input_images"]
            if len(images) != row["prompt"].count("<image>") or len(images) != 1 + turn_index:
                raise ValueError(f"image placeholder mismatch at {path}:{line_number}")
            if not row["target"].startswith("<think>"):
                raise ValueError(f"target lacks <think> at {path}:{line_number}")
            if validate_images:
                for image_path in images:
                    resolved = resolve_data_path(image_path, data_dir)
                    if not resolved.is_file():
                        raise FileNotFoundError(resolved)
            counts[(route, stage)] += 1
            rows.append(row)
    expected_counts = {
        ("direct", "decision"): 750,
        ("tool", "decision"): 750,
        ("tool", "answer_after_tool"): 750,
    }
    if dict(counts) != expected_counts or len(rows) != 2250:
        raise ValueError(f"expected 2,250 turns in three groups of 750, got {dict(counts)}")
    return rows


def encode_turn(row: dict[str, Any], processor: Any, *, max_prompt_tokens: int,
                max_response_tokens: int, data_dir: Path = ROOT / "data",
                include_pixels: bool = True) -> dict[str, Any]:
    """Mask the complete user prompt and supervise only assistant output + EOT."""
    import torch

    images = []
    for image_path in row["input_images"]:
        with Image.open(resolve_data_path(image_path, data_dir)) as opened:
            images.append(prepare_image(opened))
    image_inputs = processor.image_processor(images, return_tensors="pt")
    grid = image_inputs["image_grid_thw"]
    if len(grid) != len(images):
        raise ValueError(f"processor returned {len(grid)} grids for {len(images)} images")
    merge = int(processor.image_processor.merge_size) ** 2
    text = apply_thinking_chat_template(
        processor.tokenizer, [{"role": "user", "content": row["prompt"]}]
    )
    for image_grid in grid:
        tokens = int(image_grid.prod().item()) // merge
        if tokens <= 0:
            raise ValueError(f"empty image grid for {row['sample_id']}")
        replacement = "<|vision_start|>" + processor.image_token * tokens + "<|vision_end|>"
        text = text.replace("<image>", replacement, 1)
    if "<image>" in text:
        raise ValueError(f"unexpanded image placeholder for {row['sample_id']}")
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(text + row["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"prompt/completion tokenizer boundary changed for {row['sample_id']}")
    completion_ids = full_ids[len(prompt_ids):]
    if not prompt_ids or not completion_ids:
        raise ValueError(f"empty tokenization for {row['sample_id']}")
    if len(prompt_ids) > max_prompt_tokens or len(completion_ids) > max_response_tokens:
        raise ValueError(
            f"token limit exceeded for {row['sample_id']} turn {row['turn_index']}: "
            f"prompt={len(prompt_ids)}/{max_prompt_tokens}, "
            f"response={len(completion_ids)}/{max_response_tokens}"
        )
    ids = full_ids
    if ids.count(tokenizer.convert_tokens_to_ids(processor.image_token)) != sum(
        int(g.prod().item()) // merge for g in grid
    ):
        raise ValueError(f"vision-token count mismatch for {row['sample_id']}")
    result: dict[str, Any] = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor([-100] * len(prompt_ids) + completion_ids, dtype=torch.long),
        "prompt_length": len(prompt_ids),
        "response_length": len(completion_ids),
        "image_grid_thw": grid,
    }
    if include_pixels:
        result["pixel_values"] = image_inputs["pixel_values"]
    return result


class TurnDataset:
    def __init__(self, rows: list[dict[str, Any]], processor: Any,
                 max_prompt_tokens: int, max_response_tokens: int,
                 data_dir: Path = ROOT / "data"):
        self.rows = rows
        self.processor = processor
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return encode_turn(
            self.rows[index], self.processor,
            max_prompt_tokens=self.max_prompt_tokens,
            max_response_tokens=self.max_response_tokens,
            data_dir=self.data_dir,
        )


class Qwen3VLCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        ids = pad_sequence(
            [item["input_ids"] for item in features], batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [item["labels"] for item in features], batch_first=True, padding_value=-100
        )
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": (torch.arange(ids.shape[1])[None, :] < torch.tensor(
                [len(item["input_ids"]) for item in features]
            )[:, None]).long(),
            "pixel_values": torch.cat([item["pixel_values"] for item in features], dim=0),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in features], dim=0),
            "prompt_lengths": torch.tensor([item["prompt_length"] for item in features]),
        }
