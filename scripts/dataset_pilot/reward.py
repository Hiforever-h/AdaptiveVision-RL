"""Pure geometry only; no model/API call. Caller owns DTPO path/cost handling."""
import math

from .common import validate_box


def area(box):
    x1, y1, x2, y2 = validate_box(box)
    return (x2 - x1) * (y2 - y1)


def geometry_reward(predicted_box, reference_boxes, coverage_weight=0.5):
    """Return None for absent references (mask, not zero reward).

    predicted_box must be the actual executed crop, normalized to the original
    image. Multiple references, when supplied, must be sufficient alternatives.
    Direct answers must be handled separately by the DTPO training adapter.
    """
    if not 0 < coverage_weight < 1:
        raise ValueError("coverage_weight must be between 0 and 1")
    p = validate_box(predicted_box)
    if not reference_boxes:
        return None
    candidates = []
    for reference in reference_boxes:
        g = validate_box(reference)
        intersection = max(0, min(p[2], g[2]) - max(p[0], g[0])) * max(0, min(p[3], g[3]) - max(p[1], g[1]))
        coverage = intersection / area(g)
        iou = intersection / (area(p) + area(g) - intersection)
        reward = coverage ** coverage_weight * iou ** (1 - coverage_weight)
        candidates.append({"coverage": coverage, "iou": iou, "reward": reward})
    result = max(candidates, key=lambda c: c["reward"])
    if not math.isfinite(result["reward"]):
        raise ValueError("non-finite reward")
    return result
