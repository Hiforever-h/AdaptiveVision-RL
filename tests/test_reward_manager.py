import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import torch

    from adaptive_vision_rl.verl.reward_manager import AdaptiveVisionRewardManager
except ModuleNotFoundError:
    np = None
    torch = None
    AdaptiveVisionRewardManager = object


class FakeData:
    def __init__(self):
        self.batch = {
            "responses": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        self.non_tensor_batch = {
            "traj_uid": np.asarray(["direct"], dtype=object),
            "uid": np.asarray(["question"], dtype=object),
            "rewards": np.asarray([1.5], dtype=object),
            "episode_rewards": np.asarray([0.0], dtype=object),
            "anchor_obs": np.asarray(
                [
                    {
                        "stage": "decision",
                        "sample_id": "question",
                        "tool_reward_eligible": False,
                    }
                ],
                dtype=object,
            ),
        }

    def __len__(self):
        return 1


@unittest.skipIf(np is None or torch is None, "training dependencies are not installed")
class RewardManagerTests(unittest.TestCase):
    def test_episode_rewards_support_verl_agent_metric_reduction(self):
        config = SimpleNamespace(
            tool_advantage_coef=0.3,
            balance_penalty=0.01,
            balance_threshold=0.2,
            advantage_epsilon=1e-6,
        )
        data = FakeData()
        manager = AdaptiveVisionRewardManager(None, config)

        manager(data)

        rewards = data.non_tensor_batch["episode_rewards"]
        self.assertEqual(rewards.dtype, np.float32)
        # This is the exact reduction shape used by verl-agent metric_utils.
        reduced = rewards[np.asarray([0])].max().item()
        self.assertEqual(reduced, 1.5)


if __name__ == "__main__":
    unittest.main()
