"""Carry exactness, numeric similarity, and format through scalar step rewards."""

import math


def encode_answer_reward(*, correct: bool, score: float, format_reward: float) -> float:
    """Pack components for verl-agent's scalar environment reward transport.

    Exact answers retain the old 1 + format value. For inexact answers, format
    occupies one of five 1/8-wide buckets and similarity occupies the bucket's
    interior. The DTPO reward manager decodes the components before training.
    """
    if correct:
        return 1.0 + format_reward
    return format_reward + score / 8.0


def decode_answer_reward(value: float) -> tuple[float, float, float]:
    """Return (exact accuracy, answer score, format reward)."""
    if value >= 1.0:
        return 1.0, 1.0, max(0.0, min(0.5, value - 1.0))
    format_units = max(0, min(4, math.floor(value * 8.0 + 1e-6)))
    format_reward = format_units / 8.0
    score = max(0.0, min(0.9999, (value - format_reward) * 8.0))
    return 0.0, score, format_reward
