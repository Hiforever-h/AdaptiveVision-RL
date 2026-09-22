"""Validate exported HTML review edits and write a reviewed annotation JSONL."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import read_jsonl, write_jsonl


def checked_box(value, width: int, height: int) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("edited_reference_box_pixels must contain four coordinates")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("edited pixel coordinates must be integers")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("edited pixel box is outside the image or has no area")
    return value


def apply(dataset: Path, edits_path: Path, output: Path) -> dict:
    annotations = read_jsonl(dataset / "annotations.jsonl")
    by_id = {row["sample_id"]: row for row in annotations}
    edits = read_jsonl(edits_path)
    edit_ids = [row["sample_id"] for row in edits]
    if len(set(edit_ids)) != len(edit_ids):
        raise ValueError("Review export contains duplicate sample IDs")
    missing = sorted(set(edit_ids) - by_id.keys())
    if missing:
        raise ValueError(f"Review export contains unknown sample IDs: {missing[:5]}")
    changed = 0
    reviewed = 0
    for edit in edits:
        row = by_id[edit["sample_id"]]
        reviewed += int(bool(edit.get("verdict")))
        if edit.get("edited"):
            current_status = row["region_annotation"]["status"]
            current_box = ((row["region_annotation"].get("reference_boxes_pixels") or [None])[0])
            if edit.get("original_status") != current_status:
                raise ValueError(f"Exported original status is stale for {row['sample_id']}")
            if edit.get("original_reference_box_pixels") != current_box:
                raise ValueError(f"Exported original box is stale for {row['sample_id']}")
            status = edit.get("edited_status")
            if status not in {"localized", "global", "uncertain"}:
                raise ValueError(f"Invalid edited status for {row['sample_id']}")
            if status == "localized":
                box = checked_box(edit.get("edited_reference_box_pixels"),
                                  row["width"], row["height"])
                pixel_boxes = [box]
                normalized = [[box[0] / row["width"], box[1] / row["height"],
                               box[2] / row["width"], box[3] / row["height"]]]
            else:
                if edit.get("edited_reference_box_pixels") is not None:
                    raise ValueError(f"Non-local status must not have a box: {row['sample_id']}")
                pixel_boxes, normalized = [], []
            row["region_annotation"] = {
                **row["region_annotation"],
                "status": status,
                "reference_boxes": normalized,
                "reference_boxes_pixels": pixel_boxes,
                "coordinate_system": "human-reviewed integer xyxy pixels on original; right/bottom exclusive; normalized deterministically for reward",
            }
            row["quality_checks"] = {
                **row["quality_checks"],
                "localization_review": "human_edited",
                "region_reward_eligible": (
                    row["quality_checks"]["accepted_by_answer"] and status == "localized"),
            }
            changed += 1
        row["manual_review"] = {
            "verdict": edit.get("verdict", ""),
            "note": edit.get("note", ""),
            "review_updated_at": edit.get("updated_at"),
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "edit_source": str(edits_path),
            "original_status": edit.get("original_status"),
            "original_reference_box_pixels": edit.get("original_reference_box_pixels"),
        }
    write_jsonl(output, annotations)
    return {"records": len(annotations), "review_rows": len(edits),
            "verdict_rows": reviewed, "box_edits": changed, "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Write a new file; the active annotations.jsonl is not overwritten")
    args = parser.parse_args()
    target = args.output.resolve()
    active = args.dataset.resolve() / "annotations.jsonl"
    if target == active:
        parser.error("Refusing to overwrite active annotations.jsonl; choose a reviewed output path")
    result = apply(args.dataset.resolve(), args.edits.resolve(), target)
    print(result)
