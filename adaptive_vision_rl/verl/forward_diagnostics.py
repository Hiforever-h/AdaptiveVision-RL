"""Fixed-response replay and temporary forward controls for consistency checks."""

from __future__ import annotations

import math
from contextlib import contextmanager, ExitStack
from types import MethodType

import torch
import torch.nn.functional as F


def replay_tensors(prompts, records, *, response_width, pad_token_id):
    """Rebuild the original rollout tensors, rejecting changed inputs/positions."""

    ids, mask, positions = (prompts[key] for key in ("input_ids", "attention_mask", "position_ids"))
    if len(records) != ids.shape[0]:
        raise ValueError("Replay row count differs from the reconstructed batch")
    responses = ids.new_full((len(records), response_width), pad_token_id)
    response_mask = mask.new_zeros(responses.shape)
    log_probs = torch.zeros(responses.shape, dtype=torch.float32, device=ids.device)
    delta = torch.arange(1, response_width + 1, device=positions.device)
    # Match the rollout's positions even at masked padding, so changing padding
    # values cannot introduce a new variable in the padded-forward controls.
    response_positions = positions[..., -1:] + delta
    for row, record in enumerate(records):
        valid_prompt = mask[row].bool()
        if ids[row, valid_prompt].tolist() != record["prompt_token_ids"]:
            raise ValueError(f"Replay prompt tokens changed at row {row}")
        count = len(record["response_token_ids"])
        if not 0 < count <= response_width:
            raise ValueError(f"Replay response does not fit the original width at row {row}")
        saved_positions = torch.tensor(record["position_ids"], dtype=positions.dtype, device=positions.device)
        actual_positions = torch.cat((positions[row, ..., valid_prompt], response_positions[row, ..., :count]), -1)
        if saved_positions.shape != actual_positions.shape or not torch.equal(saved_positions, actual_positions):
            raise ValueError(f"Replay position IDs changed at row {row}")
        probabilities = record["log_probs"]["rollout"]
        if len(probabilities) != count or any(v is None or not math.isfinite(v) for v in probabilities):
            raise ValueError(f"Replay rollout log-probabilities are incomplete at row {row}")
        if not record.get("generation_prompt_alignment", {}).get("matches", False):
            raise ValueError(f"Original generation prompt was not aligned at row {row}")
        responses[row, :count] = ids.new_tensor(record["response_token_ids"])
        response_mask[row, :count] = 1
        log_probs[row, :count] = log_probs.new_tensor(probabilities)
    return {
        "prompts": ids.clone(), "responses": responses,
        "input_ids": torch.cat((ids, responses), -1),
        "attention_mask": torch.cat((mask, response_mask), -1),
        "position_ids": torch.cat((positions, response_positions), -1),
        "rollout_log_probs": log_probs,
    }


def focus_tokens(records, limit=4):
    candidates = []
    for row, record in enumerate(records):
        for item in record["largest_differences"]:
            offset = item["response_offset"]
            if record["response_token_ids"][offset] != item["token_id"]:
                raise ValueError("Replay focus token does not match the saved response")
            candidates.append({"row": row, "response_offset": offset,
                               "token_id": item["token_id"], "token_text": item["token_text"],
                               "source_probability_difference": item["probability_difference"]})
    result, seen = [], set()
    for item in sorted(candidates, key=lambda item: item["source_probability_difference"], reverse=True):
        key = item["row"], item["response_offset"]
        if key not in seen:
            result.append(item)
            seen.add(key)
        if len(result) == limit:
            break
    if not result:
        raise ValueError("Source report contains no finite focus-token comparisons")
    return result


def actor_variants(*, micro, remove_padding, temperature, diagnostics=False):
    training = dict(remove_padding=remove_padding, micro_batch_size=micro, temperature=temperature)
    single = dict(remove_padding=True, micro_batch_size=1, temperature=temperature)
    padded = dict(remove_padding=False, micro_batch_size=1, temperature=temperature)
    phases = [("actor_training", training)]
    if diagnostics:
        phases.append(("actor_training_repeat", training.copy()))
    phases.extend([("actor_packed_single", single), ("actor_padded_single", padded)])
    if diagnostics:
        phases.extend([
            ("actor_padded_batch", {**training, "remove_padding": False}),
            ("actor_base_training", {**training, "adapter": False, "temperature": 1.0}),
            ("actor_base_packed_single", {**single, "adapter": False, "temperature": 1.0}),
            ("actor_base_padded_batch", {**training, "remove_padding": False, "adapter": False, "temperature": 1.0}),
        ])
    phases.append(("actor_base", {**padded, "adapter": False, "temperature": 1.0}))
    if temperature != 1.0:
        phases.append(("actor_raw", {**training, "temperature": 1.0}))
    if diagnostics:
        phases.extend([
            ("actor_fp32_logprob", {**training, "fp32_logprob": True}),
            ("actor_fp32_head_training", {**training, "fp32_head": True}),
            ("actor_fp32_head_single", {**single, "fp32_head": True}),
            ("actor_fp32_head_padded", {**padded, "fp32_head": True}),
            ("actor_no_reduced_bf16_training", {**training, "no_reduced_bf16": True}),
            ("actor_no_reduced_bf16_single", {**single, "no_reduced_bf16": True}),
            ("actor_precise_head_training", {**training, "fp32_head": True, "no_reduced_bf16": True}),
            ("actor_precise_head_single", {**single, "fp32_head": True, "no_reduced_bf16": True}),
            ("actor_training_after_controls", training.copy()),
        ])
    return phases


def trace_reference(phase):
    if phase.startswith("actor_base"):
        return "actor_base_training"
    for prefix in ("actor_fp32_head", "actor_no_reduced_bf16", "actor_precise_head"):
        if phase.startswith(prefix) and not phase.endswith("_training"):
            return prefix + "_training"
    return "actor_training"


def extra_comparisons(values, *, temperature=1.0):
    pairs = [
        ("actor_training", "actor_training_repeat"),
        ("actor_training", "actor_training_after_controls"),
        ("actor_training", "actor_padded_batch"),
        ("actor_padded_batch", "actor_padded_single"),
        ("actor_base_training", "actor_base_packed_single"),
        ("actor_base_training", "actor_base_padded_batch"),
        ("actor_base_packed_single", "actor_base"),
        ("actor_base_padded_batch", "actor_base"),
        ("actor_training", "actor_fp32_logprob"),
        ("actor_training", "actor_fp32_head_training"),
        ("actor_training", "actor_no_reduced_bf16_training"),
        ("actor_training", "actor_precise_head_training"),
        ("actor_fp32_head_training", "actor_fp32_head_single"),
        ("actor_fp32_head_single", "actor_fp32_head_padded"),
        ("actor_no_reduced_bf16_training", "actor_no_reduced_bf16_single"),
        ("actor_precise_head_training", "actor_precise_head_single"),
    ]
    for phase in values:
        if phase.startswith(("actor_fp32_", "actor_no_reduced_bf16_", "actor_precise_head_")):
            pairs.append(("rollout", phase))
            if temperature == 1.0:
                pairs.append(("vllm_prefill", phase))
    return [(left, right) for left, right in pairs if left in values and right in values]


@contextmanager
def precision_control(module, *, fp32_head=False, no_reduced_bf16=False, logprob_module=None):
    """Temporarily change arithmetic, never parameter storage or actor config."""

    with ExitStack() as cleanup:
        if no_reduced_bf16:
            previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            cleanup.callback(setattr, torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", previous)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if logprob_module is not None:
            original = logprob_module.logprobs_from_logits

            def logprobs(logits, labels, **kwargs):
                return original(logits.float(), labels, **kwargs)

            cleanup.callback(setattr, logprob_module, "logprobs_from_logits", original)
            logprob_module.logprobs_from_logits = logprobs
        if fp32_head:
            head = module.get_output_embeddings()
            # A nested FSDP head must retain its gather/reshard wrapper.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            while isinstance(head, FSDP):
                head = head._fsdp_wrapped_module
            if type(head) is not torch.nn.Linear:
                raise TypeError("FP32 head control requires an ordinary, unquantized nn.Linear lm_head")
            if "forward" in head.__dict__:
                cleanup.callback(setattr, head, "forward", head.__dict__["forward"])
            else:
                cleanup.callback(delattr, head, "forward")

            def forward(self, hidden_states):
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    bias = None if self.bias is None else self.bias.float()
                    return F.linear(hidden_states.float(), self.weight.float(), bias)

            head.forward = MethodType(forward, head)
        yield


class ForwardTrace:
    """Keep only focus-token activations and their rows' merged image features."""

    def __init__(self, focuses):
        self.focuses = focuses
        self.values = {}
        self.dtypes = {}
        self.calls = 0
        self.selected = []
        self.current_rows = []

    def _save(self, stage, label, value):
        key = stage, label
        if key in self.values:
            raise ValueError(f"Duplicate forward capture: {key}")
        self.dtypes[key] = str(value.dtype)
        self.values[key] = value.detach().float().cpu().clone().contiguous()

    def _hidden(self, stage, value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"Unexpected hidden/logit shape at {stage}")
        for focus_index, batch_index, sequence_index in self.selected:
            self._save(stage, f"token:{focus_index}", value[batch_index, sequence_index])

    @contextmanager
    def attach(self, module, data, *, micro_batch_size, remove_padding):
        mask = data.batch["attention_mask"].bool().cpu()
        response_width = data.batch["responses"].shape[-1]
        prompt_width = mask.shape[-1] - response_width
        prompt_lengths = mask[:, :prompt_width].sum(-1).tolist()
        lengths = mask.sum(-1).tolist()
        vision = [(name, child) for name, child in module.named_modules()
                  if type(child).__name__ == "Qwen3VLVisionModel"]
        layers = [(name, child) for name, child in module.named_modules()
                  if type(child).__name__ == "Qwen3VLTextDecoderLayer"]
        if len(vision) != 1 or not layers:
            raise ValueError("Forward tracing requires the original Qwen3-VL vision and decoder layers")
        merge = vision[0][1].spatial_merge_size
        image_lengths = [int((entry["image_grid_thw"].prod(-1) // merge**2).sum())
                         for entry in data.non_tensor_batch["multi_modal_inputs"]]
        focused_rows = {item["row"] for item in self.focuses}

        def begin(_module, args, kwargs):
            start = self.calls * micro_batch_size
            end = min(start + micro_batch_size, len(lengths))
            self.calls += 1
            self.current_rows = list(range(start, end))
            self.selected = []
            for index, focus in enumerate(self.focuses):
                row, offset = focus["row"], focus["response_offset"]
                if start <= row < end:
                    if offset >= lengths[row] - prompt_lengths[row]:
                        raise ValueError("Focus token is outside the valid replay response")
                    if remove_padding:
                        batch_index = 0
                        sequence_index = sum(lengths[start:row]) + prompt_lengths[row] + offset - 1
                    else:
                        batch_index = row - start
                        sequence_index = prompt_width + offset - 1
                    self.selected.append((index, batch_index, sequence_index))

        def visual(_module, args, kwargs, output):
            image_embeds, deepstack = output
            if image_embeds.shape[0] != sum(image_lengths[row] for row in self.current_rows):
                raise ValueError("Vision feature rows differ from reconstructed image grids")
            offset = 0
            for row in self.current_rows:
                count = image_lengths[row]
                if row in focused_rows:
                    self._save("vision.merger", f"row:{row}", image_embeds[offset:offset + count])
                    for index, value in enumerate(deepstack):
                        self._save(f"vision.deepstack.{index}", f"row:{row}", value[offset:offset + count])
                offset += count

        def input_hook(stage):
            def capture(_module, args, kwargs):
                self._hidden(stage, args[0] if args else kwargs["hidden_states"])
            return capture

        with ExitStack() as cleanup:
            cleanup.callback(module.register_forward_pre_hook(begin, with_kwargs=True).remove)
            cleanup.callback(vision[0][1].register_forward_hook(visual, with_kwargs=True).remove)
            cleanup.callback(layers[0][1].register_forward_pre_hook(input_hook("text.input"), with_kwargs=True).remove)
            for index, (_name, child) in enumerate(layers):
                def capture(_module, _args, output, stage=f"text.layer.{index}"):
                    self._hidden(stage, output)
                cleanup.callback(child.register_forward_hook(capture).remove)
            head = module.get_output_embeddings()
            cleanup.callback(head.register_forward_pre_hook(input_hook("lm_head.input"), with_kwargs=True).remove)
            cleanup.callback(head.register_forward_hook(lambda _m, _a, output: self._hidden("logits", output)).remove)
            yield self
        expected_calls = math.ceil(len(lengths) / micro_batch_size)
        if self.calls != expected_calls:
            raise ValueError(f"Trace observed {self.calls} actor forwards, expected {expected_calls}")
        for index in range(len(self.focuses)):
            for stage in ("text.input", "lm_head.input", "logits", *[f"text.layer.{i}" for i in range(len(layers))]):
                if (stage, f"token:{index}") not in self.values:
                    raise ValueError(f"Missing focus activation: {stage}/token:{index}")
        for row in focused_rows:
            for stage in ("vision.merger", *[f"vision.deepstack.{i}" for i in range(len(vision[0][1].deepstack_merger_list))]):
                if (stage, f"row:{row}") not in self.values:
                    raise ValueError(f"Missing vision activation: {stage}/row:{row}")

    def compare(self, reference):
        captures, first = [], None
        missing = [list(key) for key in reference.values.keys() - self.values.keys()]
        unexpected = [list(key) for key in self.values.keys() - reference.values.keys()]
        finite = True
        def stage_order(key):
            stage = key[0]
            if stage.startswith("vision.deepstack."):
                order = 0, int(stage.rsplit(".", 1)[1])
            elif stage == "vision.merger":
                order = 1, 0
            elif stage == "text.input":
                order = 2, 0
            elif stage.startswith("text.layer."):
                order = 3, int(stage.rsplit(".", 1)[1])
            else:
                order = (4 if stage == "lm_head.input" else 5), 0
            return *order, key[1]

        for key in sorted(self.values, key=stage_order):
            value = self.values[key]
            item = {"stage": key[0], "label": key[1], "dtype": self.dtypes[key], "shape": list(value.shape)}
            other = reference.values.get(key)
            if other is not None and other.shape == value.shape:
                valid = bool(torch.isfinite(value).all() and torch.isfinite(other).all())
                finite = finite and valid
                item["all_finite"] = valid
                if valid:
                    delta = value - other
                    item.update(matches=bool(torch.equal(value, other)), max_abs_difference=float(delta.abs().max()),
                                rms_difference=float(delta.double().square().mean().sqrt()),
                                relative_l2=float(delta.double().norm() / other.double().norm().clamp_min(1e-30)))
                    if first is None and not item["matches"]:
                        first = key[0]
                if key[0] == "logits" and valid:
                    focus = self.focuses[int(key[1].split(":")[1])]
                    token = focus["token_id"]
                    top = value.topk(min(5, value.numel()))
                    item.update(sampled_token_id=token, sampled_logit=float(value[token]),
                                sampled_probability=float(value.log_softmax(-1)[token].exp()),
                                reference_sampled_logit=float(other[token]),
                                reference_sampled_probability=float(other.log_softmax(-1)[token].exp()),
                                top_token_ids=top.indices.tolist(), top_logits=top.values.tolist())
            else:
                item["shape_or_capture_mismatch"] = True
                finite = False
            captures.append(item)
        return {"forward_calls": self.calls, "capture_count": len(captures), "missing": missing,
                "unexpected": unexpected, "all_finite": finite,
                "first_nonidentical_captured_stage": first, "captures": captures}
