import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive_vision_rl.answer_reward import decode_answer_reward

try:
    import numpy as np
    from PIL import Image

    from adaptive_vision_rl.environment import AdaptiveVisionEnvironmentManager
except ModuleNotFoundError:
    np = None
    Image = None
    AdaptiveVisionEnvironmentManager = object


class _TestEnvironment(AdaptiveVisionEnvironmentManager):
    def _vision_tokens(self, image, *, cache_key=None):
        del cache_key
        return int(image.shape[0] * image.shape[1])


@unittest.skipIf(np is None, "NumPy is installed by the training environment")
class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        (root / "train/images").mkdir(parents=True)
        (root / "train/lowres").mkdir(parents=True)
        full = np.zeros((4, 4, 3), dtype=np.uint8)
        full[:2, :2] = 255
        Image.fromarray(full).save(root / "train/images/sample.png")
        Image.fromarray(full[::2, ::2]).save(root / "train/lowres/sample.png")
        config = SimpleNamespace(
            env=SimpleNamespace(
                adaptive_vision=SimpleNamespace(data_root=str(root))
            ),
            algorithm=SimpleNamespace(
                dtpo=SimpleNamespace(coverage_weight=0.5)
            ),
        )
        self.environment = _TestEnvironment(config, processor=None, is_train=True)
        self.row = {
            "sample_id": "sample",
            "question": "What is the answer?",
            "answers": ["42"],
            "image_path": "train/images/sample.png",
            "lowres_path": "train/lowres/sample.png",
            "reference_boxes": [[0.0, 0.0, 0.5, 0.5]],
            "tool_reward_eligible": True,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_direct_answer_is_one_turn_without_tool_reward(self):
        observations, _ = self.environment.reset([self.row])
        self.assertIn("Your response MUST start with <think>", observations["text"][0])
        self.assertNotIn("Example", observations["text"][0])
        observations, rewards, dones, infos = self.environment.step(
            ["<think>The answer is visible.</think><answer>42</answer>"]
        )
        self.assertTrue(dones[0])
        self.assertAlmostEqual(float(rewards[0]), 1.1)
        self.assertEqual(infos[0]["tool_calling"], 0)
        self.assertEqual(observations["text"][0].count("<image>"), 1)
        self.assertEqual(len(observations["image"][0]), 1)

    def test_direct_answer_without_think_is_invalid_and_earns_no_format_credit(self):
        self.environment.reset([self.row])
        _, rewards, dones, infos = self.environment.step(["<answer>42</answer>"])
        self.assertTrue(dones[0])
        self.assertFalse(infos[0]["is_action_valid"])
        self.assertEqual(float(rewards[0]), 1.0)
        self.assertEqual(infos[0]["format_reward"], 0.0)

    def test_close_numeric_answer_gets_partial_score_but_not_accuracy(self):
        self.environment.reset([{**self.row, "answers": ["42138"]}])
        _, rewards, dones, infos = self.environment.step(["<answer>42130</answer>"])
        accuracy, score, format_reward = decode_answer_reward(float(rewards[0]))
        self.assertTrue(dones[0])
        self.assertEqual(accuracy, 0.0)
        self.assertAlmostEqual(score, 42130 / 42138, places=5)
        self.assertEqual(format_reward, 0.0)
        self.assertEqual(infos[0]["won"], 0.0)
        self.assertAlmostEqual(infos[0]["answer_score"], score, places=5)

    def test_tool_reward_is_on_first_turn_and_second_turn_sees_two_images(self):
        observations, _ = self.environment.reset([self.row])
        self.assertEqual(len(observations["image"][0]), 1)
        call = (
            '<think>I need detail.</think><tool_call>{"name":"request_local_region",'
            '"arguments":{"bbox_2d":[0,0,500,500]}}</tool_call>'
        )
        observations, rewards, dones, infos = self.environment.step([call])
        self.assertFalse(dones[0])
        self.assertEqual(float(rewards[0]), 1.0)
        self.assertEqual(infos[0]["coverage"], 1.0)
        self.assertEqual(infos[0]["iou"], 1.0)
        self.assertEqual(len(observations["image"][0]), 2)
        self.assertEqual(
            observations["anchor"][0]["vision_tokens_step_processed"], 8
        )

        observations, rewards, dones, infos = self.environment.step(
            ["<think>The crop confirms it.</think><answer>42</answer>"]
        )
        self.assertTrue(dones[0])
        self.assertAlmostEqual(float(rewards[0]), 1.1)
        self.assertEqual(infos[0]["tool_calling"], 0)
        self.assertEqual(observations["text"][0].count("<image>"), 2)
        self.assertEqual(len(observations["image"][0]), 2)

    def test_tool_bbox_maps_to_original_pixels_and_second_image_is_crop(self):
        root = Path(self.temporary.name)
        full = np.zeros((4, 8, 3), dtype=np.uint8)
        full[:, :, 0] = np.arange(8, dtype=np.uint8)[None, :]
        full[:, :, 1] = np.arange(4, dtype=np.uint8)[:, None]
        Image.fromarray(full).save(root / "train/images/wide.png")
        low = full[::2, ::2]
        Image.fromarray(low).save(root / "train/lowres/wide.png")
        row = {
            **self.row,
            "image_path": "train/images/wide.png",
            "lowres_path": "train/lowres/wide.png",
            "tool_reward_eligible": False,
        }
        self.environment.reset([row])
        observations, _, dones, infos = self.environment.step(
            [
                '<think>Need the center.</think><tool_call>{"name":"request_local_region",'
                '"arguments":{"bbox_2d":[250,250,750,750]}}</tool_call>'
            ]
        )
        self.assertFalse(dones[0])
        self.assertEqual(infos[0]["predicted_box"], [0.25, 0.25, 0.75, 0.75])
        self.assertTrue(np.array_equal(observations["image"][0][0], low))
        self.assertTrue(np.array_equal(observations["image"][0][1], full[1:3, 2:6]))
        self.assertEqual(observations["text"][0].count("<image>"), 2)
        self.assertIn(row["question"], observations["text"][0])

    def test_tool_trajectory_averages_format_over_both_turns(self):
        self.environment.reset([self.row])
        call = (
            '<think>I need detail.</think><tool_call>{"name":"request_local_region",'
            '"arguments":{"bbox_2d":[0,0,500,500]}}</tool_call>'
        )
        _, _, dones, _ = self.environment.step([call])
        self.assertFalse(dones[0])
        _, rewards, dones, infos = self.environment.step(
            ["<answer>42</answer>"]
        )
        self.assertTrue(dones[0])
        self.assertFalse(infos[0]["is_action_valid"])
        self.assertAlmostEqual(float(rewards[0]), 1.05)
        self.assertEqual(infos[0]["format_reward"], 0.05)

    def test_mixed_batch_keeps_only_active_tool_images(self):
        direct = dict(self.row, sample_id="direct")
        tool = dict(self.row, sample_id="tool")
        self.environment.reset([direct, tool])
        call = (
            '<think>I need detail.</think><tool_call>{"name":"request_local_region",'
            '"arguments":{"bbox_2d":[0,0,500,500]}}</tool_call>'
        )

        observations, _, dones, _ = self.environment.step(
            ["<answer>42</answer>", call]
        )

        self.assertTrue(dones[0])
        self.assertFalse(dones[1])
        self.assertEqual(observations["text"][0].count("<image>"), 1)
        self.assertEqual(len(observations["image"][0]), 1)
        self.assertEqual(observations["text"][1].count("<image>"), 2)
        self.assertEqual(len(observations["image"][1]), 2)


if __name__ == "__main__":
    unittest.main()
