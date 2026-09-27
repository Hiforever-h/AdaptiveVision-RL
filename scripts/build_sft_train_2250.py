"""Select 750 tool and 750 direct trajectories for the first SFT run.

The output contains 1,500 complete trajectories and 2,250 independent turns.
Each turn has the exact prompt and image inputs used by the current stateless
AdaptiveVision environment. No images are copied into the output directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "data/visionthink_3000_300_500_balanced/train/annotations.jsonl"
TOOL_ANNOTATIONS = ROOT / "data/sft_adaptive_vision_v1/use_tool_true/annotations.jsonl"
DIRECT_ANNOTATIONS = ROOT / "data/sft_adaptive_vision_v1/use_tool_false/annotations.jsonl"
DEFAULT_OUTPUT = ROOT / "data/sft_adaptive_vision_v1/train_2250_turns"
SEED = 20260922
PER_ROUTE = 750


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def question_family(question: str) -> str:
    text = question.casefold()
    patterns = (
        ("table", r"\b(table|row|column|cell)\b"),
        ("chart", r"\b(chart|graph|plot|axis|axes|legend|bar chart|line graph)\b"),
        ("text", r"\b(read|written|word|text|label|title|caption|spell|name)\b"),
        ("count", r"\b(how many|count|number of)\b"),
        ("math", r"\b(percent|percentage|ratio|probability|average|sum|total|difference|measure|length)\b"),
    )
    for family, pattern in patterns:
        if re.search(pattern, text):
            return family
    return "other"


def area_bucket(size: list[int]) -> str:
    area = int(size[0]) * int(size[1])
    if area <= 65_536:
        return "small"
    if area <= 262_144:
        return "medium"
    if area <= 1_048_576:
        return "large"
    return "very_large"


def stratum(annotation: dict) -> tuple[str, str]:
    return question_family(annotation["question"]), area_bucket(annotation["lowres_size"])


def stratified_select(candidates: list[dict], count: int, seed: int) -> list[dict]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for annotation in candidates:
        groups[stratum(annotation)].append(annotation)

    quotas: dict[tuple[str, str], int] = {}
    remainders = []
    for key, group in groups.items():
        exact = count * len(group) / len(candidates)
        quotas[key] = math.floor(exact)
        remainders.append((exact - quotas[key], key))
    unallocated = count - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:unallocated]:
        quotas[key] += 1

    selected = []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: stable_rank(row["sample_id"], seed))
        selected.extend(ordered[: quotas[key]])
    if len(selected) != count or len({row["sample_id"] for row in selected}) != count:
        raise AssertionError("Stratified selection did not produce unique requested rows")
    return selected


def repo_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing or invalid repository-relative file: {relative}")
    return path


def qwen_box_from_pixels(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        (1000 * x1) // width,
        (1000 * y1) // height,
        (1000 * x2 + width - 1) // width,
        (1000 * y2 + height - 1) // height,
    ]


def executed_box(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        math.floor(x1 * width / 1000),
        math.floor(y1 * height / 1000),
        math.ceil(x2 * width / 1000),
        math.ceil(y2 * height / 1000),
    ]


def validate_and_flatten(annotation: dict, source: dict) -> list[dict]:
    from adaptive_vision_rl.prompts import INITIAL_PROMPT, SECOND_PROMPT
    from adaptive_vision_rl.protocol import parse_action

    sid = annotation["sample_id"]
    route = annotation["route"]
    turns = annotation["turns"]
    low_size = annotation["lowres_size"]
    original_size = annotation["original_size"]
    gold = source["answers"][0]
    if annotation["source_split"] != "train" or source["split"] != "train":
        raise ValueError(f"Non-training source: {sid}")
    if annotation["reference_answers"][0] != gold:
        raise ValueError(f"Gold answer drift: {sid}")
    if annotation["question"] != source["question"]:
        raise ValueError(f"Question drift: {sid}")
    if low_size != source["lowres_size"] or original_size != [source["width"], source["height"]]:
        raise ValueError(f"Image size drift: {sid}")
    if len(turns) != (2 if route == "tool" else 1):
        raise ValueError(f"Incorrect trajectory length: {sid}")

    first = turns[0]
    expected_prompt = INITIAL_PROMPT.format(
        question=annotation["question"], width=low_size[0], height=low_size[1]
    )
    if first["turn_index"] != 0 or first["stage"] != "decision" or first["prompt"] != expected_prompt:
        raise ValueError(f"First-turn prompt drift: {sid}")
    if first["input_images"] != [annotation["lowres_path"]]:
        raise ValueError(f"First-turn image drift: {sid}")
    parsed_first = parse_action(first["assistant"], allow_tool=True, image_size=tuple(low_size))
    if not parsed_first.valid or parsed_first.kind != ("tool" if route == "tool" else "answer"):
        raise ValueError(f"Invalid first-turn action: {sid}: {parsed_first.error}")
    if route == "direct" and parsed_first.answer != gold:
        raise ValueError(f"Direct answer differs from source gold: {sid}")

    if route == "tool":
        if source.get("source_info", {}).get("use_tool") is not True:
            raise ValueError(f"Unexpected use_tool hint for tool trajectory: {sid}")
        if not source.get("quality_checks", {}).get("region_reward_eligible"):
            raise ValueError(f"Tool trajectory lacks verified region evidence: {sid}")
        pixel_boxes = source.get("region_annotation", {}).get("reference_boxes_pixels") or []
        if not pixel_boxes or annotation["gold_bbox_original_xyxy"] != pixel_boxes[0]:
            raise ValueError(f"Source box drift: {sid}")
        qbox = qwen_box_from_pixels(pixel_boxes[0], *original_size)
        if annotation["tool_bbox_qwen_1000_xyxy"] != qbox:
            raise ValueError(f"Qwen box drift: {sid}")
        if annotation["crop_bbox_executed_original_xyxy"] != executed_box(qbox, *original_size):
            raise ValueError(f"Executed crop box drift: {sid}")
        with Image.open(repo_file(annotation["original_image_path_for_provenance"])) as original:
            expected_crop = original.convert("RGB").crop(
                tuple(annotation["crop_bbox_executed_original_xyxy"])
            )
        with Image.open(repo_file(annotation["crop_path"])) as crop:
            actual_crop = crop.convert("RGB")
            if actual_crop.size != expected_crop.size or ImageChops.difference(
                expected_crop, actual_crop
            ).getbbox() is not None:
                raise ValueError(f"Crop pixels differ from tool execution: {sid}")
        second = turns[1]
        if second["turn_index"] != 1 or second["stage"] != "answer_after_tool":
            raise ValueError(f"Second-turn metadata drift: {sid}")
        if second["prompt"] != SECOND_PROMPT.format(question=annotation["question"]):
            raise ValueError(f"Second-turn prompt drift: {sid}")
        if second["input_images"] != [annotation["lowres_path"], annotation["crop_path"]]:
            raise ValueError(f"Second-turn image drift: {sid}")
        parsed_second = parse_action(second["assistant"], allow_tool=False, image_size=tuple(low_size))
        if not parsed_second.valid or parsed_second.kind != "answer" or parsed_second.answer != gold:
            raise ValueError(f"Invalid second-turn answer: {sid}: {parsed_second.error}")
    elif route == "direct":
        if source.get("source_info", {}).get("use_tool") is not False:
            raise ValueError(f"Unexpected use_tool hint for direct trajectory: {sid}")
        if annotation.get("crop_path") is not None:
            raise ValueError(f"Direct trajectory unexpectedly has a crop: {sid}")
    else:
        raise ValueError(f"Unexpected route: {sid}: {route}")

    flat = []
    for turn in turns:
        if turn["prompt"].count("<image>") != len(turn["input_images"]):
            raise ValueError(f"Prompt/image count mismatch: {sid}")
        for path in turn["input_images"]:
            repo_file(path)
        flat.append(
            {
                "sample_id": sid,
                "route": route,
                "turn_index": turn["turn_index"],
                "stage": turn["stage"],
                "prompt": turn["prompt"],
                "input_images": turn["input_images"],
                "target": turn["assistant"],
            }
        )
    return flat


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def build(output: Path, seed: int, overwrite: bool) -> dict:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to rebuild")
    source_rows = read_jsonl(SOURCE)
    tool_rows = read_jsonl(TOOL_ANNOTATIONS)
    direct_rows = read_jsonl(DIRECT_ANNOTATIONS)
    source_by_id = {row["sample_id"]: row for row in source_rows}
    if len(source_by_id) != 3000 or len(tool_rows) != 1500 or len(direct_rows) != 1500:
        raise ValueError("Unexpected source or SFT annotation counts")

    tool_candidates = [
        row for row in tool_rows
        if row["route"] == "tool"
        and source_by_id[row["sample_id"]]["quality_checks"]["region_reward_eligible"]
    ]
    direct_candidates = [row for row in direct_rows if row["route"] == "direct"]
    selected_tool = stratified_select(tool_candidates, PER_ROUTE, seed)
    selected_direct = stratified_select(direct_candidates, PER_ROUTE, seed + 1)
    selected = sorted(
        [*selected_tool, *selected_direct],
        key=lambda row: stable_rank(row["sample_id"], seed + 2),
    )

    trajectories = []
    turns = []
    edited_thinks = 0
    for original in selected:
        annotation = copy.deepcopy(original)
        first = annotation["turns"][0]
        if annotation["route"] == "tool" and "marked region" in first["assistant"]:
            first["assistant"] = first["assistant"].replace("marked region", "relevant region")
            annotation.setdefault("sft_2250_label_edits", []).append(
                "Replaced 'marked region' with 'relevant region' because the first image has no mark."
            )
            edited_thinks += 1
        sid = annotation["sample_id"]
        turns.extend(validate_and_flatten(annotation, source_by_id[sid]))
        trajectories.append(annotation)

    if len(trajectories) != 1500 or len(turns) != 2250:
        raise AssertionError("Final trajectory or turn count is incorrect")
    if Counter(row["stage"] for row in turns) != {"decision": 1500, "answer_after_tool": 750}:
        raise AssertionError("Final stage counts are incorrect")
    if Counter(row["route"] for row in trajectories) != {"tool": 750, "direct": 750}:
        raise AssertionError("Final route counts are incorrect")
    if len({row["sample_id"] for row in trajectories}) != 1500:
        raise AssertionError("Duplicate selected sample IDs")

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "trajectories.jsonl", trajectories)
    write_jsonl(output / "turns.jsonl", turns)
    manifest = {
        "schema_version": 1,
        "purpose": "One-epoch LoRA SFT warm start for Qwen3-VL-4B-Thinking before GRPO",
        "selection_seed": seed,
        "selection": "750 region_reward_eligible tool trajectories and 750 direct trajectories; proportional strata by question family and low-resolution image area; SHA-256 ranking within strata",
        "source_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (SOURCE, TOOL_ANNOTATIONS, DIRECT_ANNOTATIONS)
        },
        "candidate_counts": {"tool_verified": len(tool_candidates), "direct": len(direct_candidates)},
        "trajectory_count": len(trajectories),
        "route_counts": dict(Counter(row["route"] for row in trajectories)),
        "turn_count": len(turns),
        "stage_counts": dict(Counter(row["stage"] for row in turns)),
        "first_turn_rationale_edits": edited_thinks,
        "selected_sample_ids": {
            "tool": sorted(row["sample_id"] for row in trajectories if row["route"] == "tool"),
            "direct": sorted(row["sample_id"] for row in trajectories if row["route"] == "direct"),
        },
        "stratum_counts": {
            route: {
                f"{family}/{area}": count
                for (family, area), count in sorted(Counter(
                    stratum(row) for row in trajectories if row["route"] == route
                ).items())
            }
            for route in ("tool", "direct")
        },
        "turn_file_sha256": sha256_file(output / "turns.jsonl"),
        "trajectory_file_sha256": sha256_file(output / "trajectories.jsonl"),
        "notes": [
            "Image paths are repository-root relative and refer to existing source images and crops.",
            "The second turn is an independent prompt with low-resolution image plus crop, matching env.history_length=0.",
            "Only prompt, input_images, and target are model inputs/labels; source hints and reference answers remain metadata.",
            "GPU-side processor token-length preflight is required before starting SFT; no image token was truncated here.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# AdaptiveVision 2,250-turn SFT training set\n\n"
        "`trajectories.jsonl` contains 1,500 complete training trajectories: "
        "750 verified one-crop trajectories and 750 direct-answer trajectories. "
        "`turns.jsonl` contains their 2,250 flattened, stateless training turns. "
        "Each turn has `sample_id`, `route`, `turn_index`, `stage`, `prompt`, "
        "`input_images`, and `target`. Image paths are relative to the repository root.\n\n"
        "The source `use_tool` flag and reference answers are never inserted into prompts. "
        "The 750 tool decisions use source regions eligible for the project tool reward. "
        "Tool first-turn rationales that referred to a nonexistent 'marked region' "
        "were corrected in this selected output; source files are unchanged.\n\n"
        "Regenerate from the repository root with:\n\n"
        "```bash\npython scripts/build_sft_train_2250.py --overwrite\n```\n\n"
        "See `manifest.json` for hashes, selected IDs, selection strata, and counts. "
        "Before SFT, verify token lengths with the exact GPU training processor.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(args.output_dir.expanduser().resolve(), args.seed, args.overwrite)
    print(json.dumps({
        "output": str(args.output_dir.expanduser().resolve()),
        "trajectories": manifest["trajectory_count"],
        "turns": manifest["turn_count"],
        "routes": manifest["route_counts"],
        "stages": manifest["stage_counts"],
        "edited_first_turn_rationales": manifest["first_turn_rationale_edits"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
