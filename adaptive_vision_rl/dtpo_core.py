"""Framework-independent DTPO reward and advantage helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import fmean, stdev
from typing import Iterable


@dataclass(frozen=True)
class TrajectoryReward:
    """One completed direct-answer or tool-use trajectory."""

    trajectory_id: str
    group_id: str
    accuracy: float
    format_reward: float
    used_tool: bool
    tool_reward: float = 0.0
    tool_reward_eligible: bool = False
    balance_reward: float = 0.0
    outcome_reward: float = 0.0
    outcome_advantage: float = 0.0
    tool_advantage: float = 0.0


def _sample_standardize(values: list[float], epsilon: float) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    deviation = stdev(values)
    if not math.isfinite(deviation) or deviation <= epsilon:
        return [0.0] * len(values)
    mean = fmean(values)
    return [(value - mean) / (deviation + epsilon) for value in values]


def assign_dtpo_rewards_and_advantages(
    records: Iterable[TrajectoryReward],
    *,
    balance_penalty: float = 0.01,
    balance_threshold: float = 0.2,
    epsilon: float = 1e-6,
) -> list[TrajectoryReward]:
    """Apply the paper-shaped balance cost and group-relative advantages.

    Outcome advantage is normalized over all trajectories for a prompt. Tool
    advantage is normalized only over eligible two-turn tool trajectories. Direct
    answers therefore receive exactly zero tool advantage, as specified for this
    project rather than AdaptVision's literal Eq. 13.
    """

    if balance_penalty < 0:
        raise ValueError("balance_penalty must be non-negative")
    if not 0 <= balance_threshold <= 1:
        raise ValueError("balance_threshold must be in [0, 1]")

    grouped: dict[str, list[TrajectoryReward]] = {}
    for record in records:
        grouped.setdefault(record.group_id, []).append(record)

    result: list[TrajectoryReward] = []
    for group in grouped.values():
        correct_direct = sum(record.accuracy > 0 and not record.used_tool for record in group)
        correct_tool = sum(record.accuracy > 0 and record.used_tool for record in group)
        denominator = correct_direct + correct_tool
        direct_correct_ratio = correct_direct / denominator if denominator else 0.0

        rewarded: list[TrajectoryReward] = []
        for record in group:
            balance = 0.0
            if record.accuracy > 0:
                if record.used_tool:
                    balance = -balance_penalty
                elif direct_correct_ratio < balance_threshold:
                    balance = -balance_penalty
            outcome = record.accuracy + record.format_reward + balance
            rewarded.append(replace(record, balance_reward=balance, outcome_reward=outcome))

        outcome_advantages = _sample_standardize(
            [record.outcome_reward for record in rewarded], epsilon
        )
        tool_positions = [
            index
            for index, record in enumerate(rewarded)
            if record.used_tool and record.tool_reward_eligible
        ]
        tool_advantages = _sample_standardize(
            [rewarded[index].tool_reward for index in tool_positions], epsilon
        )
        tool_by_position = dict(zip(tool_positions, tool_advantages, strict=True))

        for index, record in enumerate(rewarded):
            result.append(
                replace(
                    record,
                    outcome_advantage=outcome_advantages[index],
                    tool_advantage=tool_by_position.get(index, 0.0),
                )
            )
    return result


def loss_weights_for_minibatch(
    token_counts: list[int],
    is_tool_turn: list[bool],
    *,
    gradient_accumulation: int,
) -> list[float]:
    """Return per-token weights reproducing the two DTPO denominators.

    The actor divides every micro-batch loss by ``gradient_accumulation``. Giving
    each tool token ``M / N_tool`` and each answer token ``M / N_answer`` makes
    the accumulated gradient equal to ``sum(tool)/N_tool + sum(answer)/N_answer``.
    """

    if len(token_counts) != len(is_tool_turn):
        raise ValueError("token_counts and is_tool_turn must have equal length")
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    if any(count < 0 for count in token_counts):
        raise ValueError("token counts must be non-negative")

    tool_tokens = sum(count for count, tool in zip(token_counts, is_tool_turn, strict=True) if tool)
    answer_tokens = sum(count for count, tool in zip(token_counts, is_tool_turn, strict=True) if not tool)
    weights = []
    for count, tool in zip(token_counts, is_tool_turn, strict=True):
        denominator = tool_tokens if tool else answer_tokens
        weights.append(gradient_accumulation / denominator if count and denominator else 0.0)
    return weights
