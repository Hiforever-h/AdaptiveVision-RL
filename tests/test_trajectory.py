import unittest
from types import SimpleNamespace

from adaptive_vision_rl.verl.trajectory import extract_trajectory_views


class FakeData:
    def __init__(self):
        self.non_tensor_batch = {
            "traj_uid": ["direct", "tool", "tool"],
            "uid": ["question", "question", "question"],
            "rewards": [1.5, 0.25, 0.5],
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
            "rewards": [1.5, 0.5],
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


if __name__ == "__main__":
    unittest.main()
