"""DTPO advantage assignment, padding, metrics, and verl-agent driver hooks."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

from adaptive_vision_rl.dtpo_core import loss_weights_for_minibatch

from .monitoring import install_monitoring_hooks
from .trajectory import extract_trajectory_views


def pad_batch_for_dtpo(config, data, mode="copy"):
    """Pad to verl-agent divisors while masking copied rows out of DTPO entirely."""

    del mode
    from verl import DataProto

    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    rollout_divisor = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        ref_divisor = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    else:
        ref_divisor = rollout_divisor
    if "multi_modal_inputs" in data.non_tensor_batch:
        actor_divisor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    else:
        actor_divisor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    divisor = int(np.lcm.reduce(np.asarray([ref_divisor, rollout_divisor, actor_divisor])))

    batch_size = len(data)
    data.non_tensor_batch["dtpo_padding"] = np.zeros(batch_size, dtype=bool)
    remainder = batch_size % divisor
    if not remainder:
        return data

    add = divisor - remainder
    duplicate_indices = np.arange(add, dtype=np.int64) % batch_size
    duplicate = data.select_idxs(duplicate_indices)
    duplicate.non_tensor_batch["dtpo_padding"] = np.ones(add, dtype=bool)
    return DataProto.concat([data, duplicate])


def compute_dtpo_advantage(data, *, config, **_):
    """Assign turn-specific advantages and exact mini-batch DTPO loss weights."""

    response_length = data.batch["responses"].shape[-1]
    binary_mask = data.batch["attention_mask"][:, -response_length:].to(torch.float32)
    advantages = torch.zeros_like(binary_mask, dtype=torch.float32)
    views = extract_trajectory_views(data, config.algorithm.dtpo)
    coefficient = float(config.algorithm.dtpo.tool_advantage_coef)
    tool_rows: set[int] = set()

    for view in views:
        outcome = view.reward.outcome_advantage
        advantages[view.answer_row] = binary_mask[view.answer_row] * outcome
        if view.tool_row is not None:
            tool_rows.add(view.tool_row)
            combined = outcome + coefficient * view.reward.tool_advantage
            advantages[view.tool_row] = binary_mask[view.tool_row] * combined

    mini_batch_size = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    micro_batch_size = int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
    if mini_batch_size % micro_batch_size:
        raise ValueError("ppo_mini_batch_size must be divisible by ppo_micro_batch_size_per_gpu")
    if len(data) % mini_batch_size:
        raise ValueError("DTPO batch was not padded to a complete PPO mini-batch")
    accumulation = mini_batch_size // micro_batch_size
    loss_mask = torch.zeros_like(binary_mask, dtype=torch.float32)
    padding = data.non_tensor_batch.get("dtpo_padding", np.zeros(len(data), dtype=bool))

    for start in range(0, len(data), mini_batch_size):
        stop = start + mini_batch_size
        counts = [
            0 if bool(padding[row]) else int(binary_mask[row].sum().item())
            for row in range(start, stop)
        ]
        kinds = [row in tool_rows for row in range(start, stop)]
        row_weights = loss_weights_for_minibatch(
            counts,
            kinds,
            gradient_accumulation=accumulation,
        )
        for offset, weight in enumerate(row_weights):
            row = start + offset
            if weight:
                loss_mask[row] = binary_mask[row] * weight

    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages.clone()
    # The actor is switched to its loss-mask path by main_dtpo after config validation.
    # Values are normalization weights, not merely booleans.
    data.batch["loss_mask"] = loss_mask
    return data


def dtpo_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """PPO clipped loss with precomputed DTPO per-token normalization weights."""

    del loss_agg_mode
    if clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must be greater than 1")
    cliprange_low = cliprange if cliprange_low is None else cliprange_low
    cliprange_high = cliprange if cliprange_high is None else cliprange_high
    active = (response_mask > 0).to(log_prob.dtype)
    if not bool(active.any()):
        zero = log_prob.sum() * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach()

    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    losses_unclipped = -advantages * ratio
    losses_clipped = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )
    losses = torch.maximum(losses_unclipped, losses_clipped)
    dual_clip = torch.minimum(-advantages * clip_ratio_c, losses)
    losses = torch.where(advantages < 0, dual_clip, losses)

    policy_loss = torch.sum(losses * response_mask)
    clip_fraction = verl_F.masked_mean(
        torch.gt(losses_clipped, losses_unclipped).to(log_prob.dtype), active
    )
    lower_clip_fraction = verl_F.masked_mean(
        torch.gt(torch.maximum(losses_unclipped, losses_clipped), -advantages * clip_ratio_c)
        .to(log_prob.dtype)
        * (advantages < 0).to(log_prob.dtype),
        active,
    )
    ppo_kl = verl_F.masked_mean(-log_ratio, active)
    return policy_loss, clip_fraction, ppo_kl, lower_clip_fraction


def compute_dtpo_metrics(batch) -> dict[str, float]:
    if "dtpo_accuracy" not in batch.non_tensor_batch:
        return {}
    padding = batch.non_tensor_batch.get("dtpo_padding", np.zeros(len(batch), dtype=bool))
    trajectory_ids = batch.non_tensor_batch["traj_uid"]
    selected: list[int] = []
    seen: set[str] = set()
    for index, trajectory_id in enumerate(trajectory_ids):
        key = str(trajectory_id)
        if bool(padding[index]) or key in seen:
            continue
        seen.add(key)
        selected.append(index)
    if not selected:
        return {}

    def values(key):
        return np.asarray([batch.non_tensor_batch[key][i] for i in selected], dtype=float)

    accuracy = values("dtpo_accuracy")
    has_tool = values("dtpo_has_tool") > 0
    eligible = values("dtpo_tool_eligible") > 0
    tool_rewards = values("dtpo_tool_reward")
    metrics = {
        "dtpo/accuracy": float(accuracy.mean()),
        "dtpo/format_compliance": float((values("dtpo_format_reward") / 0.5).mean()),
        "dtpo/balance_reward": float(values("dtpo_balance_reward").mean()),
        "dtpo/outcome_reward": float(values("dtpo_outcome_reward").mean()),
        "dtpo/tool_call_rate": float(has_tool.mean()),
        "vision/tokens_low": float(values("vision_tokens_low").mean()),
        "vision/tokens_crop": float(values("vision_tokens_crop").mean()),
        "vision/tokens_acquired": float(values("vision_tokens_acquired").mean()),
        "vision/tokens_processed": float(values("vision_tokens_processed").mean()),
        "vision/token_ratio": float(values("vision_token_ratio").mean()),
    }
    if has_tool.any():
        metrics["dtpo/tool_answer_accuracy"] = float(accuracy[has_tool].mean())
    if (~has_tool).any():
        metrics["dtpo/direct_answer_accuracy"] = float(accuracy[~has_tool].mean())
    eligible_tool = has_tool & eligible
    if eligible_tool.any():
        metrics["dtpo/tool_reward"] = float(tool_rewards[eligible_tool].mean())
    return metrics


def install_driver_hooks(config):
    """Install the three narrow hooks required by the pinned verl-agent trainer."""

    import verl.trainer.ppo.ray_trainer as ray_trainer

    install_monitoring_hooks(ray_trainer)
    ray_trainer.compute_advantage = partial(compute_dtpo_advantage, config=config)
    ray_trainer.adjust_batch = pad_batch_for_dtpo
    if not hasattr(ray_trainer, "_adaptive_vision_original_compute_data_metrics"):
        original = ray_trainer.compute_data_metrics
        ray_trainer._adaptive_vision_original_compute_data_metrics = original

        def with_dtpo_metrics(*args, **kwargs):
            batch = kwargs.get("batch", args[0] if args else None)
            padding = batch.non_tensor_batch.get(
                "dtpo_padding", np.zeros(len(batch), dtype=bool)
            )
            metric_batch = batch
            if np.asarray(padding, dtype=bool).any():
                metric_batch = batch.select_idxs(np.flatnonzero(~np.asarray(padding, dtype=bool)))
            if "batch" in kwargs:
                original_kwargs = dict(kwargs)
                original_kwargs["batch"] = metric_batch
                metrics = original(*args, **original_kwargs)
            else:
                metrics = original(metric_batch, *args[1:], **kwargs)
            metrics.update(compute_dtpo_metrics(batch))
            return metrics

        ray_trainer.compute_data_metrics = with_dtpo_metrics
