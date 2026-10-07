import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from adaptive_vision_rl.verl.consistency import _evaluate_prefill_repeats, _markdown_report, run_verification
from adaptive_vision_rl.verl.residual_diagnostics import comparisons, focus_probabilities, saved_micro2
from scripts import verify_rollout_residual as launcher


class PrefillRepeatTests(unittest.TestCase):
    def fixture(self):
        batch = SimpleNamespace(batch={
            "prompts": torch.tensor([[0, 1, 99, 99, 2], [3, 99, 99, 99, 4]]),
            "responses": torch.tensor([[5, 6, 0], [7, 8, 9]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]]),
        })
        class Batch:
            def __init__(self):
                self.batch = batch.batch
            def __len__(self):
                return 2
        report = {"sync_events": []}
        state = SimpleNamespace(awake=False, lora_id=0, calls=[], cache_resets=0, fail=False)
        class Manager:
            def __enter__(self):
                state.awake = True
                state.lora_id += 1
                report["sync_events"].append({"lora_id": state.lora_id})
            def __exit__(self, *_):
                state.awake = False
        def reset():
            state.cache_resets += 1
        def generate(prompts, sampling_params, lora_request, use_tqdm):
            self.assertTrue(state.awake)
            self.assertEqual(lora_request.lora_int_id, state.lora_id)
            self.assertEqual(sampling_params.max_tokens, 1)
            self.assertEqual(sampling_params.prompt_logprobs, 0)
            state.calls.append(lora_request.lora_int_id)
            if state.fail and len(state.calls) == 1:
                raise RuntimeError("injected prefill failure")
            request = prompts[0]
            expanded = []
            for token in request["prompt_token_ids"]:
                expanded.extend([99] * (len(request["multi_modal_data"]["image"])+1) if token == 88 else [token])
            return [SimpleNamespace(prompt_token_ids=expanded,
                                    prompt_logprobs=[None] + [{token: SimpleNamespace(logprob=-token/10)} for token in expanded[1:]])]
        engine = SimpleNamespace(reset_prefix_cache=reset, generate=generate,
                                 llm_engine=SimpleNamespace(list_loras=lambda: [state.lora_id]))
        worker = SimpleNamespace(rollout=SimpleNamespace(inference_engine=engine), rollout_sharding_manager=Manager())
        modules = {"vllm": SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
                   "vllm.lora.request": SimpleNamespace(LoRARequest=lambda name, lora_id, path: SimpleNamespace(lora_int_id=lora_id))}
        return Batch(), worker, state, report, modules

    def test_multimodal_prompt_alignment_and_repeats_share_one_sync_then_refresh(self):
        batch, worker, state, report, modules = self.fixture()
        case, values = {"errors": {}}, {}
        with patch.dict(sys.modules, modules):
            _evaluate_prefill_repeats(worker, batch, [[1, 88, 2], [3, 88, 4]], [["one"], ["two", "three"]],
                                     case, values, report=report, persist=lambda: None)
        self.assertFalse(case["errors"])
        self.assertEqual(state.calls, [1, 1, 1, 1, 2, 2])
        self.assertEqual(state.cache_resets, 6)
        self.assertEqual(len(report["sync_events"]), 2)
        self.assertFalse(state.awake)
        expected = torch.tensor([[-.5, -.6, float("nan")], [-.7, -.8, -.9]])
        for value in values.values():
            torch.testing.assert_close(value, expected, equal_nan=True)
        for phase in values:
            self.assertTrue(all(item["matches"] for item in case[phase + "_alignment"]))

    def test_prefill_failure_exits_wake_context_and_retains_failure(self):
        batch, worker, state, report, modules = self.fixture()
        state.fail = True
        case, values = {"errors": {}}, {}
        with patch.dict(sys.modules, modules):
            _evaluate_prefill_repeats(worker, batch, [[1, 88, 2], [3, 88, 4]], [["one"], ["two", "three"]],
                                     case, values, report=report, persist=lambda: None)
        self.assertIn("injected prefill failure", case["errors"]["same_wake"])
        self.assertIn("vllm_prefill_after_wake", values)
        self.assertNotIn("vllm_prefill", values)
        self.assertFalse(state.awake)


class ResidualReportTests(unittest.TestCase):
    def record(self):
        return dict(anchor=dict(sample_id="sample"), response_token_ids=[5, 6], log_probs={
            "rollout": [-.1, -.2], "actor_old_micro2": [-.1, -1.], "actor_old_micro4": [-1., -.2]})

    def test_saved_micro2_rejects_incomplete_source_and_focus_uses_current_residual(self):
        record = self.record()
        values = saved_micro2([record], 3)
        torch.testing.assert_close(values, torch.tensor([[-.1, -1., 0.]]))
        self.assertEqual(focus_probabilities([record], limit=1)[0]["response_offset"], 1)
        record["log_probs"]["actor_old_micro2"][1] = None
        with self.assertRaisesRegex(ValueError, "complete actor_old_micro2"):
            saved_micro2([record], 3)

    def test_comparisons_separate_repeat_precision_and_no_update_ratio(self):
        values = {"rollout": torch.tensor([[-.2, -.3, float("nan")]]),
                  "actor_old_micro2": torch.tensor([[-.3, -.4, 0.]]),
                  "actor_old_micro2_no_reduced_bf16": torch.tensor([[-.25, -.35, 0.]]),
                  "vllm_prefill": torch.tensor([[-.21, -.31, float("nan")]])}
        values["actor_update_micro2"] = values["actor_old_micro2"].clone()
        values["actor_update_micro2_no_reduced_bf16"] = values["actor_old_micro2_no_reduced_bf16"].clone()
        values["vllm_prefill_same_wake_repeat"] = values["vllm_prefill"].clone()
        values["vllm_prefill_after_wake"] = values["vllm_prefill"] + .01
        diffs, ratios = comparisons(values, torch.tensor([[1, 1, 0]], dtype=torch.bool), clip_low=.2, clip_high=.24)
        self.assertEqual(diffs["vllm_prefill_vs_vllm_prefill_same_wake_repeat"]["max"], 0)
        self.assertGreater(diffs["vllm_prefill_vs_vllm_prefill_after_wake"]["max"], 0)
        self.assertEqual(diffs["rollout_vs_actor_old_micro2"]["nonfinite_tokens"], 0)
        for ratio in ratios.values():
            self.assertEqual(ratio["min"], 1)
            self.assertEqual(ratio["max"], 1)

    def test_native_crash_is_not_success_even_when_numerical_report_completed(self):
        report = dict(status="completed", actor_weights_unchanged=True, checkpoint_file_unchanged=True,
                      shutdown=dict(status="completed"))
        self.assertEqual(launcher.exit_status(0, report)["status"], "completed")
        self.assertEqual(launcher.exit_status(-11, report)["status"], "failed")
        self.assertEqual(launcher.exit_status(-11, report)["signal"], "SIGSEGV")
        self.assertEqual(launcher.exit_status(0, {})["status"], "failed")

    def test_shutdown_exception_preserves_numerical_report_but_marks_archive_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            (checkpoint / "actor").mkdir(parents=True)
            weight = checkpoint / "actor/model_world_size_1_rank_0.pt"
            weight.write_bytes(b"untouched")
            args = SimpleNamespace(output=root / "report", checkpoint=checkpoint, seed=1,
                                   residual_forward_diagnostics=True)
            def gpu(args, report, persist, *, resources):
                report.update(status="completed", actor_weights_unchanged=True)
                resources.append(object())
            with patch("adaptive_vision_rl.verl.consistency._run_gpu", side_effect=gpu), \
                 patch("adaptive_vision_rl.verl.consistency._shutdown_gpu_worker", side_effect=RuntimeError("injected shutdown failure")):
                self.assertEqual(run_verification(args), 1)
            with zipfile.ZipFile(root / "report.zip") as bundle:
                report = json.loads(bundle.read("report/report.json"))
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["shutdown"]["status"], "failed")
            self.assertEqual(weight.read_bytes(), b"untouched")

    def test_scoped_markdown_explains_historical_decode_and_actor_only_control(self):
        text = _markdown_report(dict(status="completed", residual_forward_diagnostics=True, cases=[]))
        self.assertIn("本轮不重新采样", text)
        self.assertIn("vLLM 保持原设置", text)
        self.assertIn("绝对差不能直接相加分解", text)
