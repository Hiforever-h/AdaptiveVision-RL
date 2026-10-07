import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from adaptive_vision_rl.verl.forward_diagnostics import (
    ForwardTrace,
    actor_variants,
    extra_comparisons,
    focus_tokens,
    precision_control,
    replay_tensors,
)


class FixedResponseReplayTests(unittest.TestCase):
    def setUp(self):
        self.prompts = {
            "input_ids": torch.tensor([[0, 4, 5], [6, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "position_ids": torch.tensor([[[1, 0, 1], [1, 3, 4]], [[0, 1, 2], [2, 3, 4]]]),
        }
        self.records = []
        for row, response in enumerate(([9, 2], [10, 11, 2])):
            mask = self.prompts["attention_mask"][row].bool()
            positions = self.prompts["position_ids"][row]
            self.records.append({
                "prompt_token_ids": self.prompts["input_ids"][row, mask].tolist(),
                "response_token_ids": response,
                "position_ids": torch.cat((positions[:, mask], positions[:, -1:] + torch.arange(1, len(response) + 1)), -1).tolist(),
                "log_probs": {"rollout": [-.5] * len(response)},
                "generation_prompt_alignment": {"matches": True},
                "largest_differences": [{"response_offset": 0, "token_id": response[0],
                                         "token_text": "<", "probability_difference": .1 + row}],
            })

    def test_replay_preserves_response_tokens_mask_and_all_mrope_channels(self):
        batch = replay_tensors(self.prompts, self.records, response_width=4, pad_token_id=0)
        torch.testing.assert_close(batch["responses"], torch.tensor([[9, 2, 0, 0], [10, 11, 2, 0]]))
        torch.testing.assert_close(batch["attention_mask"], torch.tensor([[0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]]))
        torch.testing.assert_close(batch["position_ids"][..., -4:], self.prompts["position_ids"][..., -1:] + torch.arange(1, 5))
        self.assertEqual(batch["rollout_log_probs"][0, 2:].tolist(), [0., 0.])
        self.assertEqual(focus_tokens(self.records, limit=1)[0]["row"], 1)

    def test_changed_inputs_or_incomplete_probabilities_are_rejected(self):
        for field in ("prompt_token_ids", "position_ids", "log_probs", "generation_prompt_alignment", "response_token_ids"):
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                if field == "prompt_token_ids":
                    records[0][field][0] += 1
                elif field == "position_ids":
                    records[0][field][1][0] += 1
                elif field == "log_probs":
                    records[0][field]["rollout"][0] = None
                elif field == "generation_prompt_alignment":
                    records[0][field]["matches"] = False
                else:
                    records[0][field] *= 3
                with self.assertRaises(ValueError):
                    replay_tensors(self.prompts, records, response_width=4, pad_token_id=0)

    def test_raw_vllm_is_not_compared_with_temperature_scaled_precision_controls(self):
        phases = dict(actor_variants(micro=4, remove_padding=True, temperature=.7, diagnostics=True))
        self.assertEqual(phases["actor_base_training"]["temperature"], 1.)
        values = {name: None for name in ["rollout", "vllm_prefill", *phases]}
        self.assertIn(("rollout", "actor_fp32_head_training"), extra_comparisons(values, temperature=.7))
        self.assertNotIn(("vllm_prefill", "actor_fp32_head_training"), extra_comparisons(values, temperature=.7))


class ForwardTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        torch.manual_seed(73)
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]}),
            vision_config=dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=4,
                               patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                               out_hidden_size=32, num_position_embeddings=64, deepstack_visual_indexes=[0, 1, 2]),
            image_token_id=120, video_token_id=119, vision_start_token_id=121, vision_end_token_id=122,
        )
        config._attn_implementation = "eager"
        cls.model = Qwen3VLForConditionalGeneration(config).eval()
        ids = torch.tensor([[0, 0, 0, 5, 121, 120, 120, 120, 120, 122, 7, 8],
                            [6, 121, 120, 120, 120, 120, 122, 121, 120, 120, 122, 7]])
        mask = ids.ne(0).long()
        grid = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 2, 4]])
        pos, _ = cls.model.model.get_rope_index(ids, image_grid_thw=grid, attention_mask=mask)
        text = (mask.cumsum(-1) - 1).masked_fill(mask.eq(0), 1)
        positions = torch.cat((text[:, None], pos.transpose(0, 1)), 1)
        records = []
        for row, response in enumerate(([9, 10, 2], [9, 10, 11, 2])):
            records.append({
                "prompt_token_ids": ids[row, mask[row].bool()].tolist(), "response_token_ids": response,
                "position_ids": torch.cat((positions[row, :, mask[row].bool()],
                                           positions[row, :, -1:] + torch.arange(1, len(response) + 1)), -1).tolist(),
                "log_probs": {"rollout": [-1.] * len(response)}, "generation_prompt_alignment": {"matches": True},
            })
        tensors = replay_tensors(dict(input_ids=ids, attention_mask=mask, position_ids=positions), records,
                                 response_width=5, pad_token_id=0)
        pixels = torch.randn(40, 24)
        cls.data = SimpleNamespace(batch=tensors, non_tensor_batch={"multi_modal_inputs": [
            dict(pixel_values=pixels[:16], image_grid_thw=grid[:1]),
            dict(pixel_values=pixels[16:], image_grid_thw=grid[1:]),
        ]})
        cls.focuses = [dict(row=0, response_offset=1, token_id=10), dict(row=1, response_offset=2, token_id=11)]

    def forward_batch(self, micro):
        outputs = []
        for start in range(0, 2, micro):
            end = start + micro
            mm = self.data.non_tensor_batch["multi_modal_inputs"][start:end]
            with torch.no_grad():
                outputs.append(self.model(
                    **{key: self.data.batch[key][start:end] for key in ("input_ids", "attention_mask")},
                    position_ids=self.data.batch["position_ids"][start:end].transpose(0, 1),
                    **{key: torch.cat([entry[key] for entry in mm]) for key in mm[0]}, use_cache=False,
                ).logits)
        return torch.cat(outputs)

    def test_real_multimodal_trace_selects_prediction_positions_across_micro_batches(self):
        baseline = self.forward_batch(2)
        traces = []
        for micro in (2, 1):
            trace = ForwardTrace(self.focuses)
            with trace.attach(self.model, self.data, micro_batch_size=micro, remove_padding=False):
                output = self.forward_batch(micro)
            torch.testing.assert_close(output, baseline, atol=1e-6, rtol=1e-5)
            for index, focus in enumerate(self.focuses):
                torch.testing.assert_close(trace.values["logits", f"token:{index}"],
                                           output[focus["row"], 12 + focus["response_offset"] - 1])
            self.assertIn(("vision.deepstack.2", "row:1"), trace.values)
            self.assertEqual(trace.calls, 2 // micro)
            traces.append(trace)
        comparison = traces[1].compare(traces[0])
        self.assertTrue(comparison["all_finite"])
        self.assertFalse(comparison["missing"] or comparison["unexpected"])
        self.assertLess(max(item["max_abs_difference"] for item in comparison["captures"]), 1e-5)

    def test_first_changed_capture_is_ordered_by_stage_instead_of_row(self):
        reference, actual = ForwardTrace([]), ForwardTrace([])
        for trace in (reference, actual):
            trace._save("text.layer.10", "token:0", torch.ones(3))
            trace._save("vision.merger", "row:0", torch.ones(2, 3))
            trace._save("vision.deepstack.2", "row:1", torch.ones(2, 3))
        for value in actual.values.values():
            value.add_(1)
        self.assertEqual(actual.compare(reference)["first_nonidentical_captured_stage"], "vision.deepstack.2")

    def test_precision_controls_restore_forward_logprob_and_global_flag_on_failure(self):
        head = self.model.get_output_embeddings()
        original = head.forward
        weights = head.weight.detach().clone()
        logprob_module = SimpleNamespace(logprobs_from_logits=lambda logits, labels: logits.dtype)
        original_logprob = logprob_module.logprobs_from_logits
        reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        hidden = torch.randn(1, 2, head.in_features, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with precision_control(self.model, fp32_head=True, no_reduced_bf16=True, logprob_module=logprob_module):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = head(hidden)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, F.linear(hidden.float(), weights.float()))
                self.assertEqual(logprob_module.logprobs_from_logits(hidden, None), torch.float32)
                self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                raise RuntimeError("injected")
        self.assertEqual(head.forward, original)
        self.assertNotIn("forward", head.__dict__)
        self.assertIs(logprob_module.logprobs_from_logits, original_logprob)
        self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, reduction)
        torch.testing.assert_close(head.weight, weights, atol=0, rtol=0)
        self.assertFalse(head._forward_hooks or head._forward_pre_hooks)

    def test_trace_hooks_are_removed_after_model_failure(self):
        trace = ForwardTrace(self.focuses)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with trace.attach(self.model, self.data, micro_batch_size=2, remove_padding=False):
                raise RuntimeError("injected")
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in self.model.modules()))
