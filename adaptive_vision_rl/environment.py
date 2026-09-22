"""Two-turn adaptive visual acquisition environment for verl-agent."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.dataset_pilot.common import answer_check, pixel_box
from scripts.dataset_pilot.reward import geometry_reward

from .protocol import ParsedAction, extract_answer_candidate, parse_action


INITIAL_PROMPT = """<image>
You are given a low-resolution version of an image and a question.

Question: {question}
Low-resolution image size: width={width}, height={height}.

Output exactly one action, with no text outside its tag:
1. Answer directly with <answer>...</answer>; or
2. Request one high-resolution crop with:
<tool_call>{{"name":"request_local_region","arguments":{{"bbox_2d":[x1,y1,x2,y2]}}}}</tool_call>

The bounding box uses xyxy coordinates normalized to the integer range 0 to 1000,
independent of the displayed image size. The origin is the top-left corner. The right
and bottom coordinates are exclusive. You may call the tool at most once. Do not output
an answer in the same turn as a tool call. You may optionally place one non-empty
reasoning block enclosed by <think> and </think> immediately before the action tag.
"""


SECOND_PROMPT = """<image>
This is the same low-resolution full image.

<image>
This is the requested high-resolution crop.

Question: {question}

Use both images. Output the final response as <answer>...</answer>, with no text outside
the tag. You may optionally place one non-empty reasoning block enclosed by <think>
and </think> immediately before the answer. You cannot call another tool.
"""


class AdaptiveVisionEnvironmentManager:
    """Vector environment driven by the per-row ``env_kwargs`` in the parquet data."""

    def __init__(self, config, processor, *, is_train: bool):
        self.config = config
        self.processor = processor
        self.is_train = is_train
        self.data_root = Path(config.env.adaptive_vision.data_root).expanduser().resolve()
        self.coverage_weight = float(config.algorithm.dtpo.coverage_weight)
        self._vision_token_cache: dict[str, int] = {}
        self._states: list[dict[str, Any]] = []

    def _path(self, relative: str) -> Path:
        path = (self.data_root / relative).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(f"dataset path escapes data_root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def _vision_tokens(self, image: np.ndarray, *, cache_key: str | None = None) -> int:
        if cache_key is not None and cache_key in self._vision_token_cache:
            return self._vision_token_cache[cache_key]
        from agent_system.multi_turn_rollout.utils import process_image

        processed = process_image(image)
        inputs = self.processor.image_processor([processed], return_tensors="pt")
        grid = inputs["image_grid_thw"]
        merge = int(self.processor.image_processor.merge_size) ** 2
        count = int(sum(int(item.prod().item()) // merge for item in grid))
        if cache_key is not None:
            self._vision_token_cache[cache_key] = count
        return count

    def _initial_observation(self, state: dict[str, Any]) -> tuple[str, list[np.ndarray], dict[str, Any]]:
        text = INITIAL_PROMPT.format(
            question=state["question"],
            width=state["low_width"],
            height=state["low_height"],
        )
        anchor = {
            "stage": "decision",
            "sample_id": state["sample_id"],
            "tool_reward_eligible": state["tool_reward_eligible"],
            "vision_tokens_low": state["vision_tokens_low"],
            "vision_tokens_crop": 0,
            "vision_tokens_step_processed": state["vision_tokens_low"],
            "vision_tokens_full": state["vision_tokens_full"],
        }
        return text, [state["low_image"]], anchor

    def reset(self, kwargs):
        rows = list(kwargs) if kwargs is not None else []
        self._states = []
        low_images_by_path: dict[Path, np.ndarray] = {}
        texts: list[str] = []
        images: list[list[np.ndarray]] = []
        anchors: list[dict[str, Any]] = []
        infos: list[dict[str, Any]] = []

        for raw in rows:
            item = dict(raw)
            answers = item.get("answers")
            if isinstance(answers, np.ndarray):
                answers = answers.tolist()
            elif isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = list(answers or [])
            reference_boxes = item.get("reference_boxes")
            if isinstance(reference_boxes, np.ndarray):
                reference_boxes = reference_boxes.tolist()
            elif not isinstance(reference_boxes, list):
                reference_boxes = list(reference_boxes or [])
            item["answers"] = [str(answer) for answer in answers]
            item["reference_boxes"] = reference_boxes
            item["tool_reward_eligible"] = bool(
                item.get("tool_reward_eligible", False) and reference_boxes
            )
            low_path = self._path(str(item["lowres_path"]))
            image_path = self._path(str(item["image_path"]))
            low_image = low_images_by_path.get(low_path)
            if low_image is None:
                low_image = self._load_rgb(low_path)
                low_images_by_path[low_path] = low_image
            full_cache_key = f"full:{image_path}"
            if full_cache_key in self._vision_token_cache:
                full_tokens = self._vision_token_cache[full_cache_key]
            else:
                full_tokens = self._vision_tokens(
                    self._load_rgb(image_path), cache_key=full_cache_key
                )
            state = {
                **item,
                "low_image": low_image,
                "low_height": int(low_image.shape[0]),
                "low_width": int(low_image.shape[1]),
                "image_path_resolved": image_path,
                "done": False,
                "stage": "decision",
                "last_images": [low_image],
                "vision_tokens_low": self._vision_tokens(low_image, cache_key=f"low:{low_path}"),
                "vision_tokens_full": full_tokens,
            }
            self._states.append(state)
            text, observation_images, anchor = self._initial_observation(state)
            texts.append(text)
            images.append(observation_images)
            anchors.append(anchor)
            infos.append({"sample_id": state["sample_id"]})

        return {"text": texts, "image": images, "anchor": anchors}, infos

    @staticmethod
    def _action_format_score(action: ParsedAction, expected_kind: str) -> float:
        """Score a valid action structure and add a bonus for real reasoning."""

        if not action.valid or action.kind != expected_kind:
            return 0.0
        return 1.0 if action.has_think else 0.5

    @staticmethod
    def _score_answer(
        answer: str | None,
        references: list[str],
        action_format_scores: list[float],
    ) -> tuple[float, float]:
        accuracy = float(answer is not None and answer_check(answer, references)["match"])
        mean_format_score = sum(action_format_scores) / len(action_format_scores)
        return accuracy, 0.5 * mean_format_score

    @staticmethod
    def _inactive_prompt(images: list[np.ndarray]) -> str:
        placeholders = "\n".join("<image>" for _ in images)
        return f"{placeholders}\nThis trajectory is already complete."

    def _execute_crop(
        self, state: dict[str, Any], bbox: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, list[float], dict[str, float] | None]:
        low_width, low_height = state["low_width"], state["low_height"]
        normalized = [
            bbox[0] / low_width,
            bbox[1] / low_height,
            bbox[2] / low_width,
            bbox[3] / low_height,
        ]
        with Image.open(state["image_path_resolved"]) as original:
            original = original.convert("RGB")
            crop_box = pixel_box(normalized, original.width, original.height)
            crop = np.asarray(original.crop(tuple(crop_box)))
            executed = [
                crop_box[0] / original.width,
                crop_box[1] / original.height,
                crop_box[2] / original.width,
                crop_box[3] / original.height,
            ]

        details = None
        if state["tool_reward_eligible"] and state.get("reference_boxes"):
            details = geometry_reward(
                executed,
                state["reference_boxes"],
                coverage_weight=self.coverage_weight,
            )
        return crop, executed, details

    def step(self, text_actions: list[str]):
        if len(text_actions) != len(self._states):
            raise ValueError("action batch size does not match environment batch size")

        next_texts: list[str] = []
        next_images: list[list[np.ndarray]] = []
        next_anchors: list[dict[str, Any]] = []
        rewards = np.zeros(len(self._states), dtype=np.float32)
        dones = np.zeros(len(self._states), dtype=bool)
        infos: list[dict[str, Any]] = []

        for index, (state, text) in enumerate(zip(self._states, text_actions, strict=True)):
            if state["done"]:
                dones[index] = True
                next_texts.append(self._inactive_prompt(state["last_images"]))
                next_images.append(state["last_images"])
                next_anchors.append({"stage": "inactive", "sample_id": state["sample_id"]})
                infos.append({"is_action_valid": True, "tool_calling": 0, "won": 0.0})
                continue

            if state["stage"] == "decision":
                action = parse_action(
                    text,
                    allow_tool=True,
                    image_size=(state["low_width"], state["low_height"]),
                )
                if action.valid and action.kind == "tool" and action.bbox is not None:
                    crop, executed, geometry = self._execute_crop(state, action.bbox)
                    tool_reward = float(geometry["reward"]) if geometry is not None else 0.0
                    tool_format_score = self._action_format_score(action, "tool")
                    crop_tokens = self._vision_tokens(crop)
                    rewards[index] = tool_reward
                    state["stage"] = "answer_after_tool"
                    state["tool_format_score"] = tool_format_score
                    state["last_images"] = [state["low_image"], crop]
                    next_texts.append(
                        SECOND_PROMPT.format(question=state["question"])
                    )
                    next_images.append(state["last_images"])
                    next_anchors.append(
                        {
                            "stage": "answer_after_tool",
                            "sample_id": state["sample_id"],
                            "tool_reward_eligible": state["tool_reward_eligible"],
                            "vision_tokens_low": state["vision_tokens_low"],
                            "vision_tokens_crop": crop_tokens,
                            "vision_tokens_step_processed": state["vision_tokens_low"] + crop_tokens,
                            "vision_tokens_full": state["vision_tokens_full"],
                        }
                    )
                    infos.append(
                        {
                            "is_action_valid": True,
                            "tool_calling": 1,
                            "won": 0.0,
                            "action_format_score": tool_format_score,
                            "predicted_box": executed,
                            "coverage": geometry["coverage"] if geometry else None,
                            "iou": geometry["iou"] if geometry else None,
                            "tool_reward": tool_reward,
                        }
                    )
                    continue

                candidate = action.answer or extract_answer_candidate(text)
                accuracy, format_reward = self._score_answer(
                    candidate,
                    state["answers"],
                    [self._action_format_score(action, "answer")],
                )
                rewards[index] = accuracy + format_reward
                state["done"] = True
                dones[index] = True
                next_texts.append(self._inactive_prompt(state["last_images"]))
                next_images.append(state["last_images"])
                next_anchors.append({"stage": "inactive", "sample_id": state["sample_id"]})
                infos.append(
                    {
                        "is_action_valid": action.valid and action.kind == "answer",
                        "tool_calling": 0,
                        "won": accuracy,
                        "format_reward": format_reward,
                        "parse_error": action.error,
                    }
                )
                continue

            action = parse_action(
                text,
                allow_tool=False,
                image_size=(state["low_width"], state["low_height"]),
            )
            candidate = action.answer or extract_answer_candidate(text)
            accuracy, format_reward = self._score_answer(
                candidate,
                state["answers"],
                [
                    float(state.get("tool_format_score", 0.0)),
                    self._action_format_score(action, "answer"),
                ],
            )
            rewards[index] = accuracy + format_reward
            state["done"] = True
            dones[index] = True
            next_texts.append(self._inactive_prompt(state["last_images"]))
            next_images.append(state["last_images"])
            next_anchors.append({"stage": "inactive", "sample_id": state["sample_id"]})
            infos.append(
                {
                    "is_action_valid": action.valid and action.kind == "answer",
                    "tool_calling": 0,
                    "won": accuracy,
                    "format_reward": format_reward,
                    "parse_error": action.error,
                }
            )

        observations = {"text": next_texts, "image": next_images, "anchor": next_anchors}
        return observations, rewards, dones, infos

    def success_evaluator(self, *, total_infos, total_batch_list, **_):
        success = defaultdict(list)
        for infos, rows in zip(total_infos, total_batch_list, strict=True):
            final_info = None
            for info, row in zip(reversed(infos), reversed(rows), strict=True):
                if row["active_masks"]:
                    final_info = info
                    break
            if final_info is None:
                raise RuntimeError("trajectory contains no active turn")
            success["success_rate"].append(float(final_info.get("won", 0.0)))
        return {key: np.asarray(values, dtype=np.float32) for key, values in success.items()}

    def close(self):
        self._states.clear()


def make_adaptive_vision_envs(config, processor):
    return (
        AdaptiveVisionEnvironmentManager(config, processor, is_train=True),
        AdaptiveVisionEnvironmentManager(config, processor, is_train=False),
    )
