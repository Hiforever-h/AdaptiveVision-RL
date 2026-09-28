import unittest
from struct import pack, unpack
from types import SimpleNamespace

from adaptive_vision_rl.answer_reward import decode_answer_reward, encode_answer_reward
from adaptive_vision_rl.verl.trajectory import extract_trajectory_views


class FakeData:
    def __init__(self):
        self.non_tensor_batch = {
            "traj_uid": ["direct", "tool", "tool"],
            "uid": ["question", "question", "question"],
            "rewards": [
                encode_answer_reward(correct=True, score=1.0, format_reward=0.1),
                0.25,
                encode_answer_reward(correct=False, score=0.0, format_reward=0.1),
            ],
            "anchor_obs": [
                {
                    "stage": "decision",
                    "sample_id": "question",
                    "tool_reward_eligible": True,
                    "vision_tokens_low": 100,
                    "vision_tokens_crop": 0,
                    "vision_tokens_step_processed": 100,
                    "vision_tokens_full": 1000,
                },
                {
                    "stage": "decision",
                    "sample_id": "question",
                    "tool_reward_eligible": True,
                    "vision_tokens_low": 100,
                    "vision_tokens_crop": 0,
                    "vision_tokens_step_processed": 100,
                    "vision_tokens_full": 1000,
                },
                {
                    "stage": "answer_after_tool",
                    "sample_id": "question",
                    "tool_reward_eligible": True,
                    "vision_tokens_low": 100,
                    "vision_tokens_crop": 50,
                    "vision_tokens_step_processed": 150,
                    "vision_tokens_full": 1000,
                },
            ],
        }

    def __len__(self):
        return 3


class FakeValidationData:
    def __init__(self):
        self.non_tensor_batch = {
            "traj_uid": ["trajectory-a", "trajectory-b"],
            # verl-agent assigns one uid to each block of env.rollout.n even
            # though validation rows are unrelated and are not repeated.
            "uid": ["framework-block", "framework-block"],
            "rewards": [
                encode_answer_reward(correct=True, score=1.0, format_reward=0.1),
                encode_answer_reward(correct=False, score=0.0, format_reward=0.0),
            ],
            "anchor_obs": [
                {
                    "stage": "decision",
                    "sample_id": "question-a",
                    "tool_reward_eligible": False,
                },
                {
                    "stage": "decision",
                    "sample_id": "question-b",
                    "tool_reward_eligible": False,
                },
            ],
        }

    def __len__(self):
        return 2


class TrajectoryTests(unittest.TestCase):
    def test_step_rows_become_direct_and_tool_trajectories(self):
        config = SimpleNamespace(
            balance_penalty=0.01,
            balance_threshold=0.2,
            advantage_epsilon=1e-6,
        )
        views = {v.reward.trajectory_id: v for v in extract_trajectory_views(FakeData(), config)}
        self.assertFalse(views["direct"].reward.used_tool)
        self.assertEqual(views["direct"].reward.accuracy, 1)
        self.assertTrue(views["tool"].reward.used_tool)
        self.assertEqual(views["tool"].reward.accuracy, 0)
        self.assertEqual(views["tool"].reward.tool_reward, 0.25)
        self.assertEqual(views["tool"].vision_tokens_acquired, 150)
        self.assertEqual(views["tool"].vision_tokens_processed, 250)

    def test_validation_groups_by_sample_instead_of_framework_block(self):
        config = SimpleNamespace(
            balance_penalty=0.01,
            balance_threshold=0.2,
            advantage_epsilon=1e-6,
        )
        views = extract_trajectory_views(FakeValidationData(), config)
        self.assertEqual([view.reward.group_id for view in views], ["question-a", "question-b"])
        self.assertEqual([view.reward.outcome_advantage for view in views], [0.0, 0.0])

    def test_partial_number_preserves_exact_accuracy_and_format(self):
        data = FakeValidationData()
        data.non_tensor_batch["rewards"][1] = encode_answer_reward(
            correct=False, score=42130 / 42138, format_reward=0.05
        )
        config = SimpleNamespace(
            balance_penalty=0.01,
            balance_threshold=0.2,
            advantage_epsilon=1e-6,
        )
        reward = extract_trajectory_views(data, config)[1].reward
        self.assertEqual(reward.accuracy, 0.0)
        self.assertAlmostEqual(reward.answer_score, 42130 / 42138)
        self.assertEqual(reward.format_reward, 0.05)
        self.assertAlmostEqual(reward.outcome_reward, 42130 / 42138 + 0.05)

    def test_nearly_exact_score_survives_float32_reward_transport(self):
        for expected_format in (0.0, 0.05, 0.1):
            with self.subTest(format_reward=expected_format):
                packed = encode_answer_reward(
                    correct=False, score=0.9999, format_reward=expected_format
                )
                stored = unpack("f", pack("f", packed))[0]
                accuracy, score, format_reward = decode_answer_reward(stored)
                self.assertEqual(accuracy, 0.0)
                self.assertEqual(format_reward, expected_format)
                self.assertAlmostEqual(score, 0.9999, places=5)

    def test_exact_answer_preserves_half_format_reward_after_float32_transport(self):
        packed = encode_answer_reward(correct=True, score=1.0, format_reward=0.05)
        stored = unpack("f", pack("f", packed))[0]
        self.assertEqual(decode_answer_reward(stored), (1.0, 1.0, 0.05))


if __name__ == "__main__":
    unittest.main()
