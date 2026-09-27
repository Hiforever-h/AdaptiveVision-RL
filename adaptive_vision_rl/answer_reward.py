"""Carry exactness, numeric similarity, and format through scalar step rewards."""

import math


FORMAT_REWARD_MAX = 0.1
_FORMAT_REWARD_UNIT = FORMAT_REWARD_MAX / 2.0
_INEXACT_SCORE_SCALE = 32.0


def encode_answer_reward(*, correct: bool, score: float, format_reward: float) -> float:
    """Pack components for verl-agent's scalar environment reward transport.

    Exact answers use 1 + format. For inexact answers, format occupies one of
    three 0.05-wide buckets and similarity occupies less than 1/32 of its
    interior. The DTPO reward manager decodes the components before training.
    """
    if correct:
        return 1.0 + format_reward
    return format_reward + score / _INEXACT_SCORE_SCALE


def decode_answer_reward(value: float) -> tuple[float, float, float]:
    """Return (exact accuracy, answer score, format reward)."""
    if value >= 1.0:
        format_units = round((value - 1.0) / _FORMAT_REWARD_UNIT)
        format_reward = max(0, min(2, format_units)) * _FORMAT_REWARD_UNIT
        return 1.0, 1.0, format_reward
    format_units = max(0, min(2, math.floor(value / _FORMAT_REWARD_UNIT + 1e-6)))
    format_reward = format_units * _FORMAT_REWARD_UNIT
    score = max(0.0, min(0.9999, (value - format_reward) * _INEXACT_SCORE_SCALE))
    return 0.0, score, format_reward
