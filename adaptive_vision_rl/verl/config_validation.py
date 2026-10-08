"""Validate batching settings before the DTPO actor is constructed."""

from __future__ import annotations


def validate_log_prob_batch_alignment(actor, rollout) -> None:
    """Prevent CLI overrides from restoring different old/update batches.

    The pinned single-GPU worker gives deprecated global micro-batch fields
    priority over their per-GPU counterparts during configuration normalization.
    Reject conflicting values before that override happens.
    """

    for config, prefix, legacy, per_gpu in (
        (actor, "actor_rollout_ref.actor", "ppo_micro_batch_size", "ppo_micro_batch_size_per_gpu"),
        (rollout, "actor_rollout_ref.rollout", "log_prob_micro_batch_size", "log_prob_micro_batch_size_per_gpu"),
    ):
        old_value = config.get(legacy)
        if old_value is not None and int(old_value) != int(config[per_gpu]):
            raise ValueError(
                f"{prefix}.{legacy}={old_value} overrides {per_gpu}={config[per_gpu]} in verl-agent. "
                f"Set {prefix}.{legacy}=null and configure {prefix}.{per_gpu} instead."
            )

    if int(actor.ppo_micro_batch_size_per_gpu) != int(rollout.log_prob_micro_batch_size_per_gpu):
        raise ValueError(
            "Old-policy log-prob and actor update micro-batches must match: "
            f"rollout.log_prob_micro_batch_size_per_gpu={rollout.log_prob_micro_batch_size_per_gpu}, "
            f"actor.ppo_micro_batch_size_per_gpu={actor.ppo_micro_batch_size_per_gpu}. "
            "Use 2 for both with the default configuration."
        )
