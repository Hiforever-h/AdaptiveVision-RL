import math
import unittest

from adaptive_vision_rl.dtpo_core import (
    TrajectoryReward,
    assign_dtpo_rewards_and_advantages,
    loss_weights_for_minibatch,
)


class DTPOCoreTests(unittest.TestCase):
    def test_correct_answer_beats_near_miss_after_tool_cost(self):
        for penalty in (0.0, 0.01, 0.2, 0.8):
            for wrong_uses_tool in (False, True):
                with self.subTest(penalty=penalty, wrong_uses_tool=wrong_uses_tool):
                    records = [
                        TrajectoryReward("exact", "q", 1, 0.1, True, answer_score=1.0),
                        TrajectoryReward("near", "q", 0, 0.1, wrong_uses_tool,
                                         answer_score=0.9999),
                    ]
                    result = {r.trajectory_id: r for r in assign_dtpo_rewards_and_advantages(
                        records, balance_penalty=penalty
                    )}
                    self.assertGreater(result["exact"].outcome_reward, result["near"].outcome_reward)
                    self.assertGreater(result["exact"].outcome_advantage, 0)
                    self.assertLess(result["near"].outcome_advantage, 0)
                    self.assertEqual(result["near"].answer_score, 0.9999)

    def test_balance_cost_and_decoupled_advantages(self):
        records = [
            TrajectoryReward("direct-ok", "q", 1, 0.1, False),
            TrajectoryReward("direct-bad", "q", 0, 0.1, False),
            TrajectoryReward("tool-low", "q", 1, 0.1, True, 0.25, True),
            TrajectoryReward("tool-high", "q", 1, 0.1, True, 1.0, True),
        ]
        result = {r.trajectory_id: r for r in assign_dtpo_rewards_and_advantages(records)}
        self.assertEqual(result["direct-ok"].balance_reward, 0)
        self.assertAlmostEqual(result["tool-low"].balance_reward, -0.01)
        self.assertAlmostEqual(result["tool-high"].outcome_reward, 1.09)
        self.assertEqual(result["direct-ok"].tool_advantage, 0)
        self.assertAlmostEqual(result["tool-low"].tool_advantage, -1 / math.sqrt(2), places=5)
        self.assertAlmostEqual(result["tool-high"].tool_advantage, 1 / math.sqrt(2), places=5)

    def test_lucky_direct_penalty_uses_group_ratio(self):
        records = [
            TrajectoryReward("direct", "q", 1, 0.1, False),
            *[
                TrajectoryReward(f"tool-{i}", "q", 1, 0.1, True, i / 5, True)
                for i in range(5)
            ],
        ]
        result = {r.trajectory_id: r for r in assign_dtpo_rewards_and_advantages(records)}
        self.assertAlmostEqual(result["direct"].balance_reward, -0.01)

    def test_loss_weights_recover_two_independent_means(self):
        weights = loss_weights_for_minibatch(
            [2, 3, 0],
            [True, False, False],
            gradient_accumulation=4,
        )
        self.assertEqual(weights[0], 2.0)
        self.assertAlmostEqual(weights[1], 4 / 3)
        self.assertEqual(weights[2], 0)
        self.assertAlmostEqual(2 * weights[0] / 4, 1.0)
        self.assertAlmostEqual(3 * weights[1] / 4, 1.0)


if __name__ == "__main__":
    unittest.main()
