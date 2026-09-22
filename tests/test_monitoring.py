import unittest

from adaptive_vision_rl.verl.monitoring import StepTimeEstimator, resolve_total_steps


class MonitoringTests(unittest.TestCase):
    def test_eta_metrics_use_smoothed_step_time(self):
        estimator = StepTimeEstimator(smoothing=0.25)
        estimator.update(100)
        estimator.update(60)

        metrics = estimator.metrics(completed_steps=2, total_steps=10)

        self.assertEqual(metrics["progress/completion_percent"], 20)
        self.assertEqual(metrics["progress/step_time_ema_seconds"], 90)
        self.assertEqual(metrics["progress/eta_seconds"], 720)
        self.assertEqual(metrics["progress/estimated_total_hours"], 0.25)

    def test_non_positive_samples_do_not_corrupt_estimate(self):
        estimator = StepTimeEstimator()
        estimator.update(0)
        estimator.update(-1)
        self.assertEqual(estimator.metrics(completed_steps=0, total_steps=10), {})

    def test_resolve_total_steps_prefers_explicit_value(self):
        config = {
            "trainer": {"total_training_steps": 12},
            "actor_rollout_ref": {
                "actor": {"optim": {"total_training_steps": 99}}
            },
        }
        self.assertEqual(resolve_total_steps(config), 12)

    def test_resolve_total_steps_uses_trainer_resolved_optimizer_value(self):
        config = {
            "trainer": {"total_training_steps": None},
            "actor_rollout_ref": {
                "actor": {"optim": {"total_training_steps": 750}}
            },
        }
        self.assertEqual(resolve_total_steps(config), 750)


if __name__ == "__main__":
    unittest.main()
