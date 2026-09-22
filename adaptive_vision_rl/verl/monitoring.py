"""WandB timing metrics and an ETA-aware progress bar for verl-agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class StepTimeEstimator:
    """Estimate remaining wall time from an exponential moving step average."""

    smoothing: float = 0.2
    smoothed_step_seconds: float | None = None

    def update(self, step_seconds: float) -> None:
        step_seconds = float(step_seconds)
        if step_seconds <= 0:
            return
        if self.smoothed_step_seconds is None:
            self.smoothed_step_seconds = step_seconds
            return
        alpha = self.smoothing
        self.smoothed_step_seconds = (
            alpha * step_seconds + (1.0 - alpha) * self.smoothed_step_seconds
        )

    def metrics(self, completed_steps: int, total_steps: int) -> dict[str, float]:
        if self.smoothed_step_seconds is None or total_steps <= 0:
            return {}
        completed = min(max(int(completed_steps), 0), int(total_steps))
        remaining = int(total_steps) - completed
        step_seconds = self.smoothed_step_seconds
        eta_seconds = remaining * step_seconds
        return {
            "progress/completion_percent": 100.0 * completed / total_steps,
            "progress/step_time_ema_seconds": step_seconds,
            "progress/steps_per_hour": 3600.0 / step_seconds,
            "progress/eta_seconds": eta_seconds,
            "progress/eta_hours": eta_seconds / 3600.0,
            "progress/estimated_total_hours": total_steps * step_seconds / 3600.0,
        }


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def resolve_total_steps(config: Mapping[str, Any]) -> int | None:
    """Read the total resolved by RayPPOTrainer, with explicit config fallback."""

    total = _nested_get(config, "trainer", "total_training_steps")
    if total is None:
        total = _nested_get(
            config,
            "actor_rollout_ref",
            "actor",
            "optim",
            "total_training_steps",
        )
    if total is None:
        return None
    total = int(total)
    return total if total > 0 else None


def install_monitoring_hooks(ray_trainer_module) -> None:
    """Install narrow process-local hooks before ``RayPPOTrainer.fit`` starts."""

    if not hasattr(ray_trainer_module, "_adaptive_vision_original_tqdm"):
        original_tqdm = ray_trainer_module.tqdm
        ray_trainer_module._adaptive_vision_original_tqdm = original_tqdm

        def eta_tqdm(*args, **kwargs):
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("mininterval", 1.0)
            kwargs.setdefault("smoothing", 0.2)
            kwargs.setdefault(
                "bar_format",
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed} elapsed, ETA {remaining}, {rate_fmt}]",
            )
            return original_tqdm(*args, **kwargs)

        ray_trainer_module.tqdm = eta_tqdm

    import verl.utils.tracking as tracking

    if hasattr(tracking, "_adaptive_vision_original_tracking"):
        return

    original_tracking = tracking.Tracking
    tracking._adaptive_vision_original_tracking = original_tracking

    class ETATracking(original_tracking):
        def __init__(self, *args, config=None, **kwargs):
            super().__init__(*args, config=config, **kwargs)
            self._adaptive_total_steps = resolve_total_steps(config or {})
            self._adaptive_step_time = StepTimeEstimator()

        def log(self, data, step, backend=None):
            payload = dict(data)
            step_seconds = payload.get("perf/time_per_step")
            if step_seconds is not None and self._adaptive_total_steps is not None:
                self._adaptive_step_time.update(float(step_seconds))
                payload.update(
                    self._adaptive_step_time.metrics(
                        completed_steps=int(step),
                        total_steps=self._adaptive_total_steps,
                    )
                )
            return super().log(payload, step, backend=backend)

    ETATracking.__name__ = "Tracking"
    tracking.Tracking = ETATracking
