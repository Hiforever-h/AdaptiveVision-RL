"""Build one-turn SFT action labels for source rows marked use_tool=true."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_vision_rl.prompts import INITIAL_PROMPT
from adaptive_vision_rl.protocol import parse_action


DATASET = ROOT / "data/visionthink_3000_300_500_balanced"
SOURCE = DATASET / "train/annotations.jsonl"
OUT = ROOT / "data/sft_adaptive_vision_v1/use_tool_true"
CROP_DIR = OUT / "crops"
OUT_REL = Path("data/sft_adaptive_vision_v1/use_tool_true")
DATASET_REL = Path("data/visionthink_3000_300_500_balanced")


GLOBAL_THINK = {
    "smart-train-27773": "There are 3 green marbles among 8 marbles, so the ratio is 3:8.",
    "smart-train-29138": "There are 4 triangles, and 4 × 3 = 12 sides in total.",
    "smart-train-29669": "The handwritten words read “Buses crawl.”",
    "smart-train-29851": "The image contains 9 squares, which is option D.",
    "smart-train-31685": "There are 10 rectangles in the top row and 1 below, for 11 total.",
    "smart-train-32196": "There are 10 cars in the top row and 5 in the second row, for 15 total.",
    "smart-train-35706": "The diagram uses decision branches and process boxes to show an algorithm, so it is a flowchart.",
    "smart-train-35792": "The target cell contains 4 patients, and the table contains 13 patients total, so the probability is 4/13.",
}


def load_source() -> list[dict]:
    with SOURCE.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def qwen_box_from_original(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        (1000 * x1) // width,
        (1000 * y1) // height,
        (1000 * x2 + width - 1) // width,
        (1000 * y2 + height - 1) // height,
    ]


def executed_box(qbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = qbox
    return [
        math.floor(x1 * width / 1000),
        math.floor(y1 * height / 1000),
        math.ceil(x2 * width / 1000),
        math.ceil(y2 * height / 1000),
    ]


def tagged(think: str, tag: str, body: str) -> str:
    return f"<think>{think}</think>\n\n<{tag}>{body}</{tag}>"


def tool_think(question: str) -> str:
    """Write a concise English action rationale shaped by the question type."""
    text = question.casefold()
    if any(term in text for term in ("table", "row", "column", "season", "player", "designation")):
        return "The relevant table entry and its row or column context are small in the overview, so I should inspect the marked region."
    if any(term in text for term in ("measure", "length", "centimeter", "centimetre", "ruler", "diameter")):
        return "The measurement marks need a closer look; I should inspect the marked region before reading the value."
    if any(term in text for term in ("percentage", "percent", "share", "score", "how many", "number", "value", "total", "average", "attendance", "cost", "price", "amount")):
        return "The exact value needed for this question is a fine detail, so I should read it in the marked high-resolution region."
    return "The identifying text or detail is small in the overview, so I should inspect the marked high-resolution region."


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = load_source()
    selected = [
        row
        for row in all_rows
        if row.get("source_info", {}).get("use_tool") is True
    ]
    if len(selected) != 1500:
        raise ValueError(f"Expected 1500 use_tool=true rows, got {len(selected)}")

    timestamp = datetime.now(timezone.utc).isoformat()
    annotations: list[dict] = []
    turns: list[dict] = []
    counts = {"tool": 0, "direct": 0}
    answer_conflicts = []

    for item in selected:
        sample_id = item["sample_id"]
        question = str(item["question"])
        low_width, low_height = item["lowres_size"]
        original_width, original_height = item["width"], item["height"]
        region = item.get("region_annotation") or {}
        region_status = str(region.get("status", "uncertain"))
        lowres_rel = (
            DATASET_REL / "train" / item["lowres_path"]
        ).as_posix()
        original_rel = (
            DATASET_REL / "train" / item["image_path"]
        ).as_posix()
        prompt = INITIAL_PROMPT.format(
            question=question,
            width=low_width,
            height=low_height,
        )
        references = [str(answer) for answer in item.get("answers", [])]

        qbox = None
        crop_box = None
        crop_rel = None
        source_box = None
        if region_status == "localized":
            boxes = region.get("reference_boxes_pixels") or []
            if not boxes:
                raise ValueError(f"Localized sample has no pixel box: {sample_id}")
            source_box = [int(value) for value in boxes[0]]
            qbox = qwen_box_from_original(
                source_box, original_width, original_height
            )
            crop_box = executed_box(qbox, original_width, original_height)
            if not (
                0 <= qbox[0] < qbox[2] <= 1000
                and 0 <= qbox[1] < qbox[3] <= 1000
            ):
                raise ValueError(f"Invalid converted Qwen box for {sample_id}: {qbox}")
            crop_rel = (OUT_REL / "crops" / f"{sample_id}.png").as_posix()
            crop_path = ROOT / crop_rel
            with Image.open(ROOT / original_rel) as original:
                original.convert("RGB").crop(tuple(crop_box)).save(
                    crop_path, format="PNG", optimize=True
                )

            think = tool_think(question)
            tool_json = json.dumps(
                {
                    "name": "request_local_region",
                    "arguments": {"bbox_2d": qbox},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            target = tagged(think, "tool_call", tool_json)
            parsed = parse_action(
                target,
                allow_tool=True,
                image_size=(low_width, low_height),
            )
            if not parsed.valid or parsed.kind != "tool":
                raise ValueError(f"Generated tool action did not parse: {sample_id}")
            route = "tool"
            counts[route] += 1
        elif region_status == "global":
            if sample_id not in GLOBAL_THINK or not references:
                raise ValueError(f"Missing direct label for global sample: {sample_id}")
            answer = references[0]
            if sample_id == "smart-train-35792":
                # Visual calculation from the displayed 2x2 table: 4 favorable
                # patients out of 4+3+4+2 = 13. Preserve the source answer below.
                answer = "4/13"
                answer_conflicts.append(
                    {
                        "sample_id": sample_id,
                        "source_answers": references,
                        "manual_target_answer": answer,
                        "reason": "The visible contingency table totals 13; the target cell is 4.",
                    }
                )
            target = tagged(GLOBAL_THINK[sample_id], "answer", answer)
            parsed = parse_action(
                target,
                allow_tool=False,
                image_size=(low_width, low_height),
            )
            if not parsed.valid or parsed.kind != "answer":
                raise ValueError(f"Generated direct action did not parse: {sample_id}")
            route = "direct"
            counts[route] += 1
        else:
            raise ValueError(
                f"Unsupported annotation status {region_status!r} for {sample_id}"
            )

        turn = {
            "turn_index": 0,
            "stage": "decision",
            "prompt_template": "INITIAL_PROMPT",
            "prompt": prompt,
            "input_images": [lowres_rel],
            "assistant": target,
        }
        annotation = {
            "sample_id": sample_id,
            "source_split": "train",
            "question": question,
            "reference_answers": references,
            "lowres_path": lowres_rel,
            "original_image_path_for_provenance": original_rel,
            "lowres_size": [low_width, low_height],
            "original_size": [original_width, original_height],
            "source_use_tool_hint": True,
            "source_region_status": region_status,
            "route": route,
            "gold_bbox_original_xyxy": source_box,
            "tool_bbox_qwen_1000_xyxy": qbox,
            "crop_bbox_executed_original_xyxy": crop_box,
            "crop_path": crop_rel,
            "region_annotation_answer": region.get("answer_from_image"),
            "turns": [turn],
            "annotation_meta": {
                "annotator": "Codex",
                "method": "One-turn action SFT from the requested use_tool=true subset; localized source boxes are used as supplied.",
                "think_language": "English",
                "think_word_limit": None,
                "created_at": timestamp,
                "coordinate_source": (
                    "region_annotation.reference_boxes_pixels"
                    if source_box is not None
                    else None
                ),
                "crop_generation": (
                    "Pillow crop using Qwen box and environment-equivalent floor/ceil conversion"
                    if crop_rel is not None
                    else None
                ),
                "source_answer_conflict": sample_id == "smart-train-35792",
            },
        }
        annotations.append(annotation)
        turns.append(
            {
                "sample_id": sample_id,
                "turn_index": 0,
                "stage": "decision",
                "prompt_template": "INITIAL_PROMPT",
                "prompt": prompt,
                "input_images": [lowres_rel],
                "target": target,
                "route": route,
                "crop_path": crop_rel,
            }
        )

    write_jsonl(OUT / "annotations.jsonl", annotations)
    write_jsonl(OUT / "turns.jsonl", turns)
    write_jsonl(OUT / "answer_conflicts.jsonl", answer_conflicts)
    manifest = {
        "source": "data/visionthink_3000_300_500_balanced/train/annotations.jsonl",
        "selection": "source_info.use_tool == true",
        "total_samples": len(annotations),
        "route_counts": counts,
        "crop_count": sum(row["crop_path"] is not None for row in annotations),
        "turn_count": len(turns),
        "turn_policy": "one first-turn action per sample; no second-turn answer targets",
        "coordinate_system": "Qwen-VL normalized xyxy in [0,1000] on the low-resolution full image",
        "output_root_relative_to_repo": OUT_REL.as_posix(),
        "created_at": timestamp,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    build()
