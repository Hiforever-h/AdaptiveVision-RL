"""A bounded fresh training run with local, per-step diagnostic metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported metric type: {type(value).__name__}")


class ShortRunRecorder:
    def __init__(self, output, *, steps, scheduler_horizon):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output / "metrics.jsonl"
        # Refuse reusing an earlier run: it could falsely appear to have 10 steps.
        self.metrics_path.touch(exist_ok=False)
        self.state = dict(status="running", start_step=0, requested_steps=steps,
                          stop_step=steps, recorded_steps=[], scheduler_horizon=scheduler_horizon,
                          scheduler_horizon_preserved=True, initialized_from="sft_base_new_lora",
                          old_logprob_micro_batch_size=2, update_micro_batch_size=2,
                          checkpoint_saving=False, validation=False)
        self.persist()

    def persist(self):
        path = self.output / "run_state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def record(self, data, step):
        clean = _json_value(data)
        nonfinite = [str(key) for key, value in data.items() if value is not None and clean[key] is None]
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(dict(step=int(step), metrics=clean, nonfinite_metrics=nonfinite),
                                    ensure_ascii=False, allow_nan=False) + "\n")
        if nonfinite:
            self.state.setdefault("nonfinite_metrics", {})[str(step)] = nonfinite
        if "training/global_step" in data:
            self.state["recorded_steps"].append(int(step))
            self.persist()

    def finish(self):
        expected = list(range(1, self.state["requested_steps"] + 1))
        if self.state["recorded_steps"] != expected:
            self.fail(f"Expected training steps {expected}; received {self.state['recorded_steps']}")
            raise RuntimeError(self.state["error"])
        self.state["status"] = "completed"
        self.persist()

    def fail(self, error):
        self.state.update(status="failed", error=str(error))
        self.persist()


def install_short_run(trainer, *, steps, output, tracking_module=None):
    """Install after init_workers so optimizer/warmup keep the full-run horizon."""
    config = trainer.config
    if steps != 10:
        raise ValueError("This diagnostic is limited to exactly 10 fresh training steps")
    if config.trainer.resume_mode != "disable" or config.trainer.resume_from_path is not None:
        raise ValueError("Short run must start fresh from SFT, with automatic resume disabled")
    if (config.trainer.save_freq > 0 or config.trainer.test_freq > 0
            or config.trainer.val_before_train or config.trainer.get("val_only", False)):
        raise ValueError("Short run must disable checkpoint saving and validation")
    if (int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu) != 2
            or int(config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu) != 2):
        raise ValueError("Short run requires old-policy and update micro-batches both equal to 2")
    horizon = int(config.actor_rollout_ref.actor.optim.total_training_steps)
    if horizon < steps or len(trainer.train_dataloader) < steps:
        raise ValueError("Original training schedule/dataloader contains fewer than 10 steps")
    recorder = ShortRunRecorder(output, steps=steps, scheduler_horizon=horizon)
    optim = config.actor_rollout_ref.actor.optim
    warmup = int(optim.get("lr_warmup_steps", -1))
    recorder.state["warmup_steps"] = warmup if warmup >= 0 else int(horizon * float(optim.get("lr_warmup_steps_ratio", 0)))
    recorder.persist()
    # Workers already own a scheduler initialized with `horizon`. Change only
    # the driver's stopping condition; do not initialize a 10-step scheduler.
    trainer.total_training_steps = steps
    config.trainer.total_training_steps = steps
    if tracking_module is None:
        import verl.utils.tracking as tracking_module
    original_tracking = tracking_module.Tracking

    class DiagnosticTracking(original_tracking):
        def log(self, data, step, backend=None):
            recorder.record(data, step)
            return super().log(data, step, backend=backend)

    tracking_module.Tracking = DiagnosticTracking

    def restore():
        tracking_module.Tracking = original_tracking

    return recorder, restore
