"""Reward manager exposing DTPO components to verl-agent."""

from __future__ import annotations

import numpy as np
import torch

from .trajectory import extract_trajectory_views


class AdaptiveVisionRewardManager:
    def __init__(self, tokenizer, dtpo_config, num_examine: int = 0):
        self.tokenizer = tokenizer
        self.dtpo_config = dtpo_config
        self.num_examine = num_examine

    def __call__(self, data, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        views = extract_trajectory_views(data, self.dtpo_config)
        row_count = len(data)
        extras = {
            "dtpo_accuracy": [0.0] * row_count,
            "dtpo_format_reward": [0.0] * row_count,
            "dtpo_balance_reward": [0.0] * row_count,
            "dtpo_outcome_reward": [0.0] * row_count,
            "dtpo_tool_reward": [0.0] * row_count,
            "dtpo_has_tool": [0.0] * row_count,
            "dtpo_tool_eligible": [0.0] * row_count,
            "vision_tokens_low": [0.0] * row_count,
            "vision_tokens_crop": [0.0] * row_count,
            "vision_tokens_acquired": [0.0] * row_count,
            "vision_tokens_processed": [0.0] * row_count,
            "vision_tokens_full": [0.0] * row_count,
            "vision_token_ratio": [0.0] * row_count,
        }

        attention_mask = data.batch["attention_mask"]
        response_length = data.batch["responses"].shape[-1]
        response_mask = attention_mask[:, -response_length:]
        tool_coefficient = float(self.dtpo_config.tool_advantage_coef)
        # verl-agent's metric_utils applies NumPy reductions followed by
        # ``.item()`` to this non-tensor field. An object array containing
        # Python floats returns a Python float from ``max()``, which has no
        # ``.item()``. Build a numeric array explicitly instead of mutating the
        # object array produced by the trajectory collector.
        episode_rewards = np.zeros(row_count, dtype=np.float32)

        for view in views:
            record = view.reward
            ratio = (
                view.vision_tokens_acquired / view.vision_tokens_full
                if view.vision_tokens_full
                else 0.0
            )
            for row in view.row_indices:
                extras["dtpo_accuracy"][row] = record.accuracy
                extras["dtpo_format_reward"][row] = record.format_reward
                extras["dtpo_balance_reward"][row] = record.balance_reward
                extras["dtpo_outcome_reward"][row] = record.outcome_reward
                extras["dtpo_tool_reward"][row] = record.tool_reward
                extras["dtpo_has_tool"][row] = float(record.used_tool)
                extras["dtpo_tool_eligible"][row] = float(record.tool_reward_eligible)
                extras["vision_tokens_low"][row] = view.vision_tokens_low
                extras["vision_tokens_crop"][row] = view.vision_tokens_crop
                extras["vision_tokens_acquired"][row] = view.vision_tokens_acquired
                extras["vision_tokens_processed"][row] = view.vision_tokens_processed
                extras["vision_tokens_full"][row] = view.vision_tokens_full
                extras["vision_token_ratio"][row] = ratio
                episode_rewards[row] = (
                    record.outcome_reward + tool_coefficient * record.tool_reward
                )

            answer_length = int(response_mask[view.answer_row].sum().item())
            if answer_length:
                reward_tensor[view.answer_row, answer_length - 1] = record.outcome_reward
            if view.tool_row is not None:
                tool_length = int(response_mask[view.tool_row].sum().item())
                if tool_length:
                    reward_tensor[view.tool_row, tool_length - 1] = (
                        tool_coefficient * record.tool_reward
                    )

        data.non_tensor_batch["episode_rewards"] = episode_rewards

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extras}
        return reward_tensor
