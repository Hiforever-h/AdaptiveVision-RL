import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from adaptive_vision_rl.verl.consistency import (
    _markdown_report,
    _reference_actions,
    object_array,
    probability_difference,
    response_prompt_log_probs,
    run_verification,
    tensor_fingerprint,
    token_alignment,
)


class ConsistencyReportTests(unittest.TestCase):
    def test_controlled_crop_uses_protocol_coordinates_instead_of_image_pixels(self):
        from adaptive_vision_rl.protocol import parse_action

        rows = [{"env_kwargs": {"reference_boxes": [[.25, .25, .75, .75]]}},
                {"env_kwargs": {"reference_boxes": []}}]
        for text in _reference_actions(rows):
            action = parse_action(text, allow_tool=True, image_size=(1600, 1200))
            self.assertTrue(action.valid)
            self.assertEqual(action.bbox, (400., 300., 1200., 900.))

    def test_probability_metrics_ignore_padding_but_report_missing_valid_tokens(self):
        left = torch.tensor([[.8, .4, .9, .3]]).log()
        right = torch.tensor([[.6, .5, .1, float("nan")]]).log()
        mask = torch.tensor([[True, True, False, True]])
        result = probability_difference(left, right, mask)
        self.assertEqual(result["requested_tokens"], 3)
        self.assertEqual(result["finite_tokens"], 2)
        self.assertEqual(result["nonfinite_tokens"], 1)
        self.assertAlmostEqual(result["mean"], .15, places=6)
        self.assertAlmostEqual(result["max"], .2, places=6)
        self.assertAlmostEqual(result["std"], 2 ** .5 * .05, places=6)

    def test_teacher_forcing_requires_exact_expanded_prompt_alignment(self):
        expected = [10, 99, 99, 11, 12, 13]
        output = SimpleNamespace(
            prompt_token_ids=expected,
            prompt_logprobs=[None] + [{token: SimpleNamespace(logprob=-i)} for i, token in enumerate(expected[1:], 1)],
        )
        values, alignment = response_prompt_log_probs(output, expected, 4, 2)
        torch.testing.assert_close(values, torch.tensor([-4., -5.]))
        self.assertTrue(alignment["matches"])
        output.prompt_token_ids = [10, 99, 11, 12, 13]
        values, alignment = response_prompt_log_probs(output, expected, 4, 2)
        self.assertFalse(alignment["matches"])
        self.assertEqual(alignment["available_response_logprobs"], 0)
        self.assertTrue(torch.isnan(values).all())

    def test_incomplete_prompt_logprob_is_retained_as_missing(self):
        output = SimpleNamespace(prompt_token_ids=[1, 2, 3], prompt_logprobs=[None, {2: SimpleNamespace(logprob=-1)}, {}])
        values, alignment = response_prompt_log_probs(output, [1, 2, 3], 1, 2)
        self.assertTrue(alignment["matches"])
        self.assertEqual(alignment["available_response_logprobs"], 1)
        self.assertEqual(values[0], -1)
        self.assertTrue(torch.isnan(values[1]))

    def test_fingerprint_includes_names_and_actual_bfloat16_storage(self):
        state = {"layer.lora_A.weight": torch.ones(1, 2, dtype=torch.bfloat16),
                 "layer.lora_B.weight": torch.zeros(2, 1, dtype=torch.bfloat16)}
        first = tensor_fingerprint(state)
        self.assertTrue(first["all_finite"])
        self.assertEqual(first["nonzero_b_elements"], 0)
        self.assertEqual(first, tensor_fingerprint(dict(reversed(list(state.items())))))
        state["layer.lora_B.weight"][0, 0] = .5
        second = tensor_fingerprint(state)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(second["nonzero_b_elements"], 1)

    def test_ragged_equal_length_rows_remain_one_dimensional_objects(self):
        rows = object_array([[1, 2], [3, 4]])
        self.assertEqual(rows.shape, (2,))
        self.assertEqual(rows[0], [1, 2])
        self.assertFalse(token_alignment([1, 2], [1])["matches"])

    def test_failure_still_packages_report_and_preserves_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "global_step_300"
            (checkpoint / "actor").mkdir(parents=True)
            original = checkpoint / "actor/model_world_size_1_rank_0.pt"
            original.write_bytes(b"checkpoint must remain untouched")
            args = SimpleNamespace(output=root / "report", checkpoint=checkpoint, seed=1)
            with patch("adaptive_vision_rl.verl.consistency._run_gpu", side_effect=RuntimeError("injected diagnostic failure")):
                code = run_verification(args)
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertIn("injected diagnostic failure", report["fatal_error"])
            self.assertTrue(report["checkpoint_file_unchanged"])
            self.assertEqual(original.read_bytes(), b"checkpoint must remain untouched")
            with zipfile.ZipFile(root / "report.zip") as archive:
                self.assertIn("report/REPORT.md", archive.namelist())
                self.assertIn("report/run.log", archive.namelist())
                self.assertFalse(any(name.endswith(".pt") for name in archive.namelist()))

    def test_output_inside_checkpoint_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            args = SimpleNamespace(output=checkpoint / "report", checkpoint=checkpoint)
            with self.assertRaisesRegex(ValueError, "outside"):
                run_verification(args)
            self.assertFalse(checkpoint.exists())

    def test_replay_report_identifies_saved_rollout_and_current_actor_computations(self):
        focus = {"row": 1, "response_offset": 21, "token_id": 27, "probability": .25}
        report = {
            "status": "completed", "checkpoint": "/checkpoint", "forward_diagnostics": True,
            "replay": {"source_report": "/original/report.json"},
            "cases": [{"name": "decision", "errors": {}, "comparisons": {}, "forward_diagnostics": {
                "all_captures_valid": True, "trace_file": "decision_forward_trace.json",
                "focus_tokens": [focus],
                "first_nonidentical_captured_stages": {"actor_training_repeat": None},
                "phases": {"actor_training_repeat": {"reference_phase": "actor_training", "focus_probabilities": [focus]}},
            }}],
        }
        text = _markdown_report(report)
        self.assertIn("`rollout` 概率沿用原报告", text)
        self.assertIn("actor 与 vllm_prefill 使用当前代码重算", text)
        self.assertIn("row 1 / response 21 / ID 27", text)
        self.assertIn("| actor_training_repeat | actor_training | 记录值完全相同 | 0.250000 |", text)
        self.assertIn("重放时首次为 vLLM 固定 token 重算前", text)
        self.assertIn("不代表已证明该模块是根因", text)
