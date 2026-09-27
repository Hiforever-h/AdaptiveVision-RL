import hashlib
import json
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/pilot20"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def validate_box(box) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("box must contain four coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
        raise ValueError("coordinates must be numbers")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in box):
        raise ValueError("coordinates must be finite and within [0, 1]")
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box must have positive area")
    return [float(v) for v in box]


def pixel_box(box, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = validate_box(box)
    return [math.floor(x1 * width), math.floor(y1 * height),
            math.ceil(x2 * width), math.ceil(y2 * height)]


def normalize_answer(answer: str) -> str:
    # Preserve units, percentages and mathematical signs; no fuzzy acceptance.
    return " ".join(unicodedata.normalize("NFKC", answer).casefold().split()).rstrip(".。")


def numeric_value(answer: str):
    text = normalize_answer(answer)
    pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[+-]?\d+)?"
    if not re.fullmatch(pattern, text):
        return None
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def answer_check(prediction: str, answers: list[str]) -> dict:
    if any(normalize_answer(prediction) == normalize_answer(a) for a in answers):
        return {"match": True, "method": "normalized_exact"}
    number = numeric_value(prediction)
    if number is not None and any(number == numeric_value(a) for a in answers):
        return {"match": True, "method": "numeric_exact"}
    return {"match": False, "method": "unmatched_requires_review"}


def answer_score(prediction: str, answers: list[str]) -> float:
    """Exact answers score 1; close pure numbers receive relative partial credit.

    Keep ``answer_check`` strict for correctness metrics and annotation filtering.
    Non-numeric answers, numbers with units, and opposite signs get no partial
    credit. The cap keeps an inexact number distinguishable from an exact match.
    """
    if answer_check(prediction, answers)["match"]:
        return 1.0
    predicted = numeric_value(prediction)
    if predicted is None:
        return 0.0
    scores = []
    for answer in answers:
        expected = numeric_value(answer)
        if expected is None or predicted * expected <= 0:
            continue
        relative_error = abs(predicted - expected) / max(abs(predicted), abs(expected))
        scores.append(max(0.0, 1.0 - float(relative_error)))
    return min(0.9999, max(scores, default=0.0))
