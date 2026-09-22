"""Strict parsing for the one-tool, at-most-two-turn interaction protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal


_DIRECT_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)
_TOOL_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*"
    r"<tool_call>(?P<call>.*?)</tool_call>\s*$",
    re.DOTALL,
)
_ANSWER_ANYWHERE_RE = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    kind: Literal["answer", "tool", "invalid"]
    valid: bool
    answer: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    error: str | None = None


def _nonempty(value: str) -> bool:
    return bool(value and value.strip())


def _unwrap_boxed(value: str) -> str:
    """Unwrap the common ``\boxed{...}`` answer form without interpreting LaTex."""

    value = value.strip()
    for prefix in (r"\boxed{", r"\fbox{"):
        if value.startswith(prefix) and value.endswith("}"):
            return value[len(prefix) : -1].strip()
    return value


def extract_answer_candidate(text: str) -> str | None:
    """Extract answer content even when the rest of the format is invalid.

    Accuracy and format are separate rewards. A missing ``<think>`` block therefore
    loses format reward without automatically losing answer accuracy.
    """

    if not isinstance(text, str):
        return None
    matches = list(_ANSWER_ANYWHERE_RE.finditer(text))
    if len(matches) != 1:
        return None
    answer = _unwrap_boxed(matches[0].group("answer"))
    return answer or None


def parse_action(text: str, *, allow_tool: bool, image_size: tuple[int, int]) -> ParsedAction:
    """Parse a complete model turn.

    Tool coordinates use Qwen-VL's native 0--1000 normalized ``xyxy`` space and
    are converted to the displayed low-resolution image here. Right and bottom
    are exclusive. A non-empty ``think`` block is required, and extra text outside
    the action tags is rejected so the format reward has an unambiguous definition.
    """

    if not isinstance(text, str):
        return ParsedAction("invalid", False, error="response is not text")

    direct = _DIRECT_RE.fullmatch(text)
    if direct:
        if not _nonempty(direct.group("think")):
            return ParsedAction("invalid", False, error="empty think block")
        answer = _unwrap_boxed(direct.group("answer"))
        if not answer:
            return ParsedAction("invalid", False, error="empty answer block")
        return ParsedAction("answer", True, answer=answer)

    tool = _TOOL_RE.fullmatch(text)
    if not tool:
        return ParsedAction("invalid", False, error="response does not match answer or tool schema")
    if not allow_tool:
        return ParsedAction("invalid", False, error="a second tool call is not allowed")
    if not _nonempty(tool.group("think")):
        return ParsedAction("invalid", False, error="empty think block")

    try:
        payload = json.loads(tool.group("call"))
    except json.JSONDecodeError as exc:
        return ParsedAction("invalid", False, error=f"invalid tool JSON: {exc.msg}")

    if not isinstance(payload, dict) or payload.get("name") != "request_local_region":
        return ParsedAction("invalid", False, error="unexpected tool name")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != {"bbox_2d"}:
        return ParsedAction("invalid", False, error="tool arguments must contain only bbox_2d")
    bbox = arguments["bbox_2d"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return ParsedAction("invalid", False, error="bbox_2d must contain four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return ParsedAction("invalid", False, error="bbox coordinates must be numeric")

    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return ParsedAction("invalid", False, error="bbox is outside the 0-1000 coordinate space")
    width, height = image_size
    image_bbox = (
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        x2 * width / 1000.0,
        y2 * height / 1000.0,
    )
    return ParsedAction("tool", True, bbox=image_bbox)
