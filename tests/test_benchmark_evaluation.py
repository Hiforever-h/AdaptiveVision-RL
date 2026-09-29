import tempfile
import unittest
import json
import sys
import io
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import evaluate_benchmarks as benchmark


class FakeEvaluator:
    processor = object()

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompts, images):
        self.calls.append((prompts, images))
        self.last_image_token_counts = [
            [10 if image.width < 100 else 20 for image in group]
            for group in images
        ]
        return next(self.responses)


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_mode_must_select_exactly_one_workflow(self):
        parser = benchmark.build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--base-model", "model", "--output-dir", "results"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--base-model", "model", "--output-dir", "results", "--mode", "both"])
        self.assertEqual(parser.parse_args(["--base-model", "model", "--output-dir", "results",
                                            "--mode", "low_tool"]).mode, "low_tool")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        high, low = root / "high.png", root / "low.png"
        Image.new("RGB", (200, 100), "white").save(high)
        Image.new("RGB", (100, 50), "white").save(low)
        self.samples = [
            benchmark.BenchmarkSample(f"ChartQA:{i}", "ChartQA", "How many?", ["3"], high, low, None)
            for i in range(3)
        ]

    def tearDown(self):
        self.temp.cleanup()

    def test_one_crop_then_answer_and_invalid_second_call(self):
        tool = ('<think>Need zoom.</think><tool_call>{"name":"request_local_region",'
                '"arguments":{"bbox_2d":[250,250,750,750]}}</tool_call>')
        evaluator = FakeEvaluator([
            ["<think>Visible.</think><answer>3</answer>", tool, tool],
            ["<think>Zoomed.</think><answer>3</answer>", tool],
        ])
        with patch.object(benchmark, "full_image_token_count", return_value=40):
            records, _ = benchmark.evaluate_batch(evaluator, self.samples, "low_tool")
        self.assertEqual(len(evaluator.calls), 2)
        self.assertEqual([len(group) for group in evaluator.calls[1][1]], [2, 2])
        self.assertEqual([r["correct"] for r in records], [True, True, False])
        self.assertEqual([r["vision_tokens_low_second"] for r in records], [0, 20, 20])
        self.assertEqual(records[1]["vision_ratio"], (20 + 20) / 40)
        self.assertEqual(records[1]["vision_ratio_true"], (20 + 20 + 20) / 40)
        self.assertIsNone(records[2]["prediction"])
        self.assertFalse(records[2]["final_answer_valid"])

    def test_high_only_never_executes_tool(self):
        tool = ('<think>Need zoom.</think><tool_call>{"name":"request_local_region",'
                '"arguments":{"bbox_2d":[0,0,500,500]}}</tool_call>')
        evaluator = FakeEvaluator([[tool, "<think>Visible.</think><answer>3</answer>"]])
        records, _ = benchmark.evaluate_batch(evaluator, self.samples[:2], "high_only")
        self.assertEqual(len(evaluator.calls), 1)
        self.assertFalse(records[0]["correct"])
        self.assertFalse(records[0]["used_tool"])
        self.assertTrue(records[1]["correct"])
        self.assertEqual(records[1]["vision_ratio_true"], 1.0)

    def test_scoring_and_micro_accuracy(self):
        sample = self.samples[0]
        self.assertEqual(benchmark.score_answer(sample, "3"), (True, "normalized_exact"))
        chart = benchmark.BenchmarkSample(sample.eval_id, "ChartQA", "q", ["100"], sample.high_path, sample.low_path, None)
        self.assertTrue(benchmark.score_answer(chart, "104")[0])
        self.assertFalse(benchmark.score_answer(chart, "106")[0])
        math = benchmark.BenchmarkSample(sample.eval_id, "MathVerse", "q", ["B"], sample.high_path, sample.low_path, "multi-choice")
        self.assertTrue(benchmark.score_answer(math, "option B")[0])
        self.assertFalse(benchmark.score_answer(math, "option C")[0])
        block = benchmark.metric_block([{"correct": True, "used_tool": False, "format_compliance": 1,
                                          "first_action_valid": True, "final_answer_valid": True,
                                          "vision_tokens_low_first": 1, "vision_tokens_low_second": 0,
                                          "vision_tokens_crop": 0, "vision_tokens_full": 2,
                                          "vision_tokens_acquired": 1, "vision_tokens_processed": 1,
                                          "vision_ratio": .5, "vision_ratio_true": .5}])
        self.assertEqual(block["accuracy"], 1.0)

    def test_cli_evaluates_only_requested_model_variant(self):
        root = Path(self.temp.name)
        selection = root / "selection.json"
        selection.write_text("{}")
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        for extra_args, expected_base, expected_adapter in (
            ([], True, None),
            (["--adapter", str(adapter)], False, str(adapter)),
        ):
            output = root / ("base_output" if expected_base else "adapter_output")
            argv = ["evaluate_benchmarks.py", "--base-model", "replaceable-model",
                    "--output-dir", str(output), "--mode", "high_only", "--limit", "1"] + extra_args
            fake = FakeEvaluator([["<think>Visible.</think><answer>3</answer>"]])
            with patch.object(sys, "argv", argv), \
                 patch.object(benchmark.sys, "platform", "linux"), \
                 patch.object(benchmark, "load_benchmark_samples", return_value=(self.samples[:1], selection)), \
                 patch.object(benchmark, "resolve_lora_adapter", return_value=adapter) as resolve, \
                 patch.object(benchmark, "VLLMEvaluator", return_value=fake) as evaluator:
                benchmark.main()
            self.assertEqual(resolve.called, not expected_base)
            self.assertEqual(evaluator.call_args.args[0].model, "replaceable-model")
            self.assertEqual(evaluator.call_args.args[0].base_model, expected_base)
            self.assertEqual(evaluator.call_args.args[1], adapter if not expected_base else None)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["adapter_path"], expected_adapter)
            self.assertEqual(summary["metrics"]["high_only"]["overall"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
