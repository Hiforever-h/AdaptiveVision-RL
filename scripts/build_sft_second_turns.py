"""Assemble manually authored second-turn labels for use_tool=true SFT rows.

Second-turn <think> text must be authored per sample in
data/sft_adaptive_vision_v1/use_tool_true/second_turn_thinks.jsonl. This script
does not generate or paraphrase any reasoning text; it only restores crops,
copies the source gold answer verbatim, assembles the conversation records, and
validates their schema and paths.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_vision_rl.prompts import SECOND_PROMPT
from adaptive_vision_rl.protocol import parse_action


OUT = ROOT / "data/sft_adaptive_vision_v1/use_tool_true"
SOURCE = ROOT / "data/visionthink_3000_300_500_balanced/train/annotations.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def regenerate_crop(annotation: dict) -> tuple[str, list[int]]:
    """Replay the stored Qwen box against the original image as the env does."""
    crop_rel = annotation.get("crop_path")
    original_rel = annotation.get("original_image_path_for_provenance")
    qbox = annotation.get("tool_bbox_qwen_1000_xyxy")
    if not crop_rel or not original_rel or not qbox:
        raise ValueError(f"Tool sample lacks crop provenance: {annotation['sample_id']}")

    original_path = ROOT / original_rel
    crop_path = ROOT / crop_rel
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    original_width, original_height = annotation["original_size"]
    x1, y1, x2, y2 = qbox
    crop_box = [
        math.floor(x1 * original_width / 1000),
        math.floor(y1 * original_height / 1000),
        math.ceil(x2 * original_width / 1000),
        math.ceil(y2 * original_height / 1000),
    ]
    with Image.open(original_path) as image:
        image = image.convert("RGB")
        if image.size != (original_width, original_height):
            raise ValueError(f"Original image size changed: {annotation['sample_id']}")
        crop = image.crop(tuple(crop_box))
        if crop.width <= 0 or crop.height <= 0:
            raise ValueError(f"Empty crop for {annotation['sample_id']}: {crop_box}")
        crop.save(crop_path, format="PNG", optimize=True)
    if crop_box != annotation.get("crop_bbox_executed_original_xyxy"):
        raise ValueError(f"Crop geometry changed for {annotation['sample_id']}: {crop_box}")
    return crop_rel, crop_box


def build() -> None:
    annotations_path = OUT / "annotations.jsonl"
    think_path = OUT / "second_turn_thinks.jsonl"
    annotations = read_jsonl(annotations_path)
    source_rows = read_jsonl(SOURCE)
    manual_thinks = read_jsonl(think_path)
    source_by_id = {row["sample_id"]: row for row in source_rows}

    if len(annotations) != 1500:
        raise ValueError(f"Expected 1,500 annotations, got {len(annotations)}")
    think_by_id: dict[str, str] = {}
    for row in manual_thinks:
        sample_id = row.get("sample_id")
        think = row.get("think")
        if not sample_id or not isinstance(think, str) or not think.strip():
            raise ValueError(f"Invalid manual think record: {row}")
        if sample_id in think_by_id:
            raise ValueError(f"Duplicate manual think for {sample_id}")
        think_by_id[sample_id] = think

    tool_ids = {row["sample_id"] for row in annotations if row["route"] == "tool"}
    missing = sorted(tool_ids - think_by_id.keys())
    unexpected = sorted(think_by_id.keys() - tool_ids)
    if missing or unexpected:
        raise ValueError(
            f"Manual think coverage must exactly match tool samples; "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )

    flat_turns: list[dict] = []
    second_turns: list[dict] = []
    crop_count = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    for annotation in annotations:
        sample_id = annotation["sample_id"]
        annotation.setdefault("annotation_meta", {}).update(
            {
                "method": (
                    "First-turn action supervision plus individually authored "
                    "answer-after-crop supervision for localized tool trajectories."
                ),
                "think_language": "English",
                "think_word_limit": None,
            }
        )
        source = source_by_id.get(sample_id)
        if source is None:
            raise ValueError(f"Missing source record for {sample_id}")
        turns = annotation.get("turns") or []
        if not turns or turns[0].get("turn_index") != 0:
            raise ValueError(f"Missing decision turn for {sample_id}")
        decision_turn = turns[0]
        flat_turns.append(
            {
                "sample_id": sample_id,
                "turn_index": 0,
                "stage": "decision",
                "prompt_template": decision_turn["prompt_template"],
                "prompt": decision_turn["prompt"],
                "input_images": decision_turn["input_images"],
                "target": decision_turn["assistant"],
                "route": annotation["route"],
                "crop_path": annotation.get("crop_path"),
            }
        )
        if annotation["route"] != "tool":
            annotation["turns"] = [decision_turn]
            continue

        crop_rel, _ = regenerate_crop(annotation)
        crop_count += 1
        answers = source.get("answers") or []
        if not answers or not isinstance(answers[0], str) or not answers[0]:
            raise ValueError(f"Missing string gold answer for {sample_id}")
        gold_answer = answers[0]
        think = think_by_id[sample_id]
        if "<think>" in think or "</think>" in think or "<answer>" in think or "<tool_call>" in think:
            raise ValueError(f"Manual think includes protocol tags for {sample_id}")

        prompt = SECOND_PROMPT.format(question=annotation["question"])
        target = f"<think>{think}</think>\n\n<answer>{gold_answer}</answer>"
        parsed = parse_action(
            target,
            allow_tool=False,
            image_size=tuple(annotation["lowres_size"]),
        )
        if not parsed.valid or parsed.kind != "answer":
            raise ValueError(f"Invalid second-turn target for {sample_id}: {parsed.error}")
        if parsed.answer != gold_answer:
            raise ValueError(f"Serialized answer differs from gold for {sample_id}")

        second_turn = {
            "turn_index": 1,
            "stage": "answer_after_tool",
            "prompt_template": "SECOND_PROMPT",
            "prompt": prompt,
            "input_images": [annotation["lowres_path"], crop_rel],
            "assistant": target,
            "answer_target_source": "source annotations.jsonl: answers[0] (verbatim)",
            "think_author": "Codex, individually authored",
        }
        annotation["turns"] = [decision_turn, second_turn]
        annotation.setdefault("annotation_meta", {}).update(
            {
                "second_turn_annotator": "Codex",
                "second_turn_method": "Individually authored per sample; answer copied verbatim from source answers[0].",
                "second_turn_created_at": timestamp,
            }
        )
        flat_row = {
            "sample_id": sample_id,
            "turn_index": 1,
            "stage": "answer_after_tool",
            "prompt_template": "SECOND_PROMPT",
            "prompt": prompt,
            "input_images": [annotation["lowres_path"], crop_rel],
            "target": target,
            "route": "tool",
            "crop_path": crop_rel,
            "answer_target_source": "source annotations.jsonl: answers[0] (verbatim)",
        }
        flat_turns.append(flat_row)
        second_turns.append(flat_row)

    if crop_count != 1492 or len(second_turns) != 1492:
        raise ValueError(
            f"Expected 1,492 crops and second turns, got {crop_count}/{len(second_turns)}"
        )

    write_jsonl(annotations_path, annotations)
    write_jsonl(OUT / "turns.jsonl", flat_turns)
    write_jsonl(OUT / "second_turns.jsonl", second_turns)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "turn_count": len(flat_turns),
            "second_turn_count": len(second_turns),
            "crop_count": crop_count,
            "turn_policy": (
                "localized tool trajectories have a decision turn and an answer-after-crop turn; "
                "global direct-answer samples have one turn"
            ),
            "second_turn_answer_source": "source annotations.jsonl answers[0], copied verbatim",
            "second_turn_think_method": "per-sample authored by Codex; no template or rule-based generation",
            "second_turn_created_at": timestamp,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total_annotations": len(annotations),
                "tool_trajectories": len(second_turns),
                "crops_regenerated": crop_count,
                "second_turns": len(second_turns),
                "total_turns": len(flat_turns),
                "manual_thinks": len(think_by_id),
                "created_at": timestamp,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    build()
