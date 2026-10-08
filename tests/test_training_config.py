import unittest

from adaptive_vision_rl.verl.config_validation import validate_log_prob_batch_alignment


class Config(dict):
    def __getattr__(self, name):
        return self[name]


class TrainingBatchConfigTests(unittest.TestCase):
    def config(self, *, update=2, old=2, legacy_update=None, legacy_old=None):
        return (
            Config(ppo_micro_batch_size_per_gpu=update, ppo_micro_batch_size=legacy_update),
            Config(log_prob_micro_batch_size_per_gpu=old, log_prob_micro_batch_size=legacy_old),
        )

    def test_accepts_matching_default_and_joint_batch_changes(self):
        for micro in (1, 2, 4):
            with self.subTest(micro=micro):
                validate_log_prob_batch_alignment(*self.config(update=micro, old=micro))

    def test_rejects_mismatched_old_and_update_batches(self):
        for update, old in ((2, 4), (4, 2)):
            with self.subTest(update=update, old=old):
                with self.assertRaisesRegex(ValueError, "micro-batches must match"):
                    validate_log_prob_batch_alignment(*self.config(update=update, old=old))

    def test_rejects_legacy_fields_that_silently_override_per_gpu(self):
        for overrides, field in (
            ({"legacy_old": 4}, "rollout.log_prob_micro_batch_size=4"),
            ({"legacy_update": 4}, "actor.ppo_micro_batch_size=4"),
            ({"legacy_update": 4, "legacy_old": 4}, "actor.ppo_micro_batch_size=4"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, field):
                    validate_log_prob_batch_alignment(*self.config(**overrides))

    def test_accepts_legacy_fields_when_they_do_not_change_effective_batches(self):
        validate_log_prob_batch_alignment(*self.config(legacy_update=2, legacy_old=2))


if __name__ == "__main__":
    unittest.main()
