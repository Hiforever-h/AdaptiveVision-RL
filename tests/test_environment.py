import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.environment.reset([self.row])
        _, rewards, dones, infos = self.environment.step(
            ["<think>The answer is visible.</think><answer>42</answer>"]
        )
        self.assertTrue(dones[0])
        self.assertEqual(float(rewards[0]), 1.5)
        self.assertEqual(infos[0]["tool_calling"], 0)

    def test_tool_reward_is_on_first_turn_and_second_turn_sees_two_images(self):
        observations, _ = self.environment.reset([self.row])
        self.assertEqual(len(observations["image"][0]), 1)
        call = (
            '<think>I need detail.</think><tool_call>{"name":"request_local_region",'
            '"arguments":{"bbox_2d":[0,0,1,1]}}</tool_call>'
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

        _, rewards, dones, infos = self.environment.step(
            ["<think>The crop confirms it.</think><answer>42</answer>"]
        )
        self.assertTrue(dones[0])
        self.assertEqual(float(rewards[0]), 1.5)
        self.assertEqual(infos[0]["tool_calling"], 0)


if __name__ == "__main__":
    unittest.main()
