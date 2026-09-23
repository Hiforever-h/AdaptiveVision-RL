import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from adaptive_vision_rl.protocol import parse_action
from scripts.evaluate_dtpo import (
    EvalSample,
    action_format_score,
    evaluate_batch,
    execute_crop,
    metric_block,
    resolve_lora_adapter,
)


class EvaluationTests(unittest.TestCase):
    def test_resolves_latest_verl_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "global_step_42/actor/lora_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(json.dumps({"r": 64}))
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "latest_checkpointed_iteration.txt").write_text("42\n")
            self.assertEqual(resolve_lora_adapter(root), adapter.resolve())

    def test_resolver_falls_back_to_highest_complete_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (2, 10):
                adapter = root / f"global_step_{step}/actor/lora_adapter"
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text(json.dumps({"r": 64}))
                (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            expected = root / "global_step_10/actor/lora_adapter"
            self.assertEqual(resolve_lora_adapter(root), expected.resolve())

    def test_crop_uses_same_normalized_coordinate_mapping_as_environment(self):
        image = Image.new("RGB", (800, 400))
        action = parse_action(
            '<tool_call>{"name":"request_local_region","arguments":'
            '{"bbox_2d":[250,250,750,750]}}</tool_call>',
            allow_tool=True,
            image_size=(400, 200),
        )
        crop, executed = execute_crop(image, (400, 200), action.bbox)
        self.assertEqual(crop.size, (400, 200))
        self.assertEqual(executed, [0.25, 0.25, 0.75, 0.75])

    def test_format_score_matches_training_semantics(self):
        plain = parse_action(
            "<answer>42</answer>", allow_tool=True, image_size=(400, 200)
        )
        reasoned = parse_action(
            "<think>Visible in the table.</think><answer>42</answer>",
            allow_tool=True,
            image_size=(400, 200),
        )
        self.assertEqual(action_format_score(plain, "answer"), 0.5)
        self.assertEqual(action_format_score(reasoned, "answer"), 1.0)

    def test_metrics_separate_direct_and_tool_accuracy(self):
        base = {
            "first_action_valid": True,
            "final_answer_valid": True,
            "format_compliance": 1.0,
            "outcome_reward": 1.5,
            "tool_reward_eligible": False,
            "tool_reward": None,
            "vision_tokens_low": 10,
            "vision_tokens_crop": 0,
            "vision_tokens_acquired": 10,
            "vision_tokens_processed": 10,
            "vision_token_ratio": 0.5,
            "estimated_generation_seconds": 0.1,
        }
        records = [
            {**base, "correct": True, "used_tool": False},
            {
                **base,
                "correct": False,
                "used_tool": True,
                "vision_tokens_crop": 5,
            },
        ]
        metrics = metric_block(records)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["direct_answer_accuracy"], 1.0)
        self.assertEqual(metrics["tool_answer_accuracy"], 0.0)
        self.assertEqual(metrics["tool_call_rate"], 0.5)

    def test_batch_evaluation_runs_direct_and_tool_routes(self):
        class FakeGrid:
            def __init__(self, value):
                self.value = value

            def prod(self):
                return self

            def item(self):
                return self.value

        class FakeImageProcessor:
            merge_size = 1

            def __call__(self, images, return_tensors):
                self.assertEqual(return_tensors, "pt")
                return {
                    "image_grid_thw": [
                        FakeGrid(image.width * image.height) for image in images
                    ]
                }

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError(f"{left!r} != {right!r}")

        class FakeProcessor:
            image_processor = FakeImageProcessor()

        class FakeEvaluator:
            processor = FakeProcessor()

            def generate(self, texts, images):
                if all(len(group) == 1 for group in images):
                    return [
                        "<answer>42</answer>",
                        '<tool_call>{"name":"request_local_region","arguments":'
                        '{"bbox_2d":[0,0,500,500]}}</tool_call>',
                    ]
                return ["<answer>42</answer>"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "full.png"
            lowres_path = root / "low.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            Image.new("RGB", (4, 4), "white").save(lowres_path)
            samples = [
                EvalSample(
                    sample_id=f"sample-{index}",
                    question=f"Question {index}",
                    answers=["42"],
                    image_path=image_path,
                    lowres_path=lowres_path,
                    reference_boxes=[[0.0, 0.0, 0.5, 0.5]],
                    tool_reward_eligible=True,
                    source_use_tool=bool(index),
                )
                for index in range(2)
            ]
            records, _ = evaluate_batch(
                FakeEvaluator(), samples, coverage_weight=0.5
            )

        self.assertEqual([record["used_tool"] for record in records], [False, True])
        self.assertTrue(all(record["correct"] for record in records))
        self.assertEqual(records[1]["predicted_box"], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(records[1]["tool_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
