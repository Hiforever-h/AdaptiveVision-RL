"""Read-only, single-GPU comparison of the actual DTPO actor and vLLM paths.

CUDA/verl imports stay inside the runner so report and alignment checks can be
tested on CPU. This module never calls an optimizer or checkpoint writer.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from adaptive_vision_rl.verl.forward_diagnostics import (
    ForwardTrace, actor_variants, extra_comparisons, focus_tokens,
    precision_control, replay_tensors, trace_reference,
)


def object_array(items):
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def token_alignment(expected, actual):
    expected, actual = list(expected), list(actual or [])
    mismatches = [i for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    if len(expected) != len(actual):
        mismatches.append(min(len(expected), len(actual)))
    first = mismatches[0] if mismatches else None
    return {
        "matches": first is None,
        "expected_length": len(expected),
        "actual_length": len(actual),
        "first_mismatch": first,
        "expected_context": expected[max(0, first - 4):first + 5] if first is not None else [],
        "actual_context": actual[max(0, first - 4):first + 5] if first is not None else [],
    }


def response_prompt_log_probs(output, expected_ids, prompt_length, response_length):
    """Do not silently align logprobs using a different multimodal prompt."""

    alignment = token_alignment(expected_ids, output.prompt_token_ids)
    values = torch.full((response_length,), float("nan"), dtype=torch.float32)
    entries = output.prompt_logprobs
    if alignment["matches"] and entries is not None and len(entries) == len(expected_ids):
        for offset in range(response_length):
            index = prompt_length + offset
            entry = entries[index]
            token_id = expected_ids[index]
            if entry is not None and token_id in entry:
                values[offset] = float(entry[token_id].logprob)
    alignment["available_response_logprobs"] = int(torch.isfinite(values).sum())
    return values, alignment


def probability_difference(left, right, mask):
    left, right = left.detach().float().cpu(), right.detach().float().cpu()
    mask = mask.detach().bool().cpu()
    if left.shape != right.shape or left.shape != mask.shape:
        raise ValueError("log-probabilities and mask must have the same shape")
    valid = mask & torch.isfinite(left) & torch.isfinite(right)
    result = {
        "requested_tokens": int(mask.sum()),
        "finite_tokens": int(valid.sum()),
        "nonfinite_tokens": int((mask & ~valid).sum()),
    }
    if not valid.any():
        return result
    diff = (left[valid].exp() - right[valid].exp()).abs()
    logdiff = (left[valid] - right[valid]).abs()
    result.update(
        mean=float(diff.mean()),
        std=float(diff.std(correction=1)) if len(diff) > 1 else 0.0,
        max=float(diff.max()),
        p50=float(diff.quantile(.5)),
        p95=float(diff.quantile(.95)),
        p99=float(diff.quantile(.99)),
        mean_abs_logprob_difference=float(logdiff.mean()),
        max_abs_logprob_difference=float(logdiff.max()),
        probability_difference_gt_001=float((diff > .01).float().mean()),
        probability_difference_gt_005=float((diff > .05).float().mean()),
        probability_difference_gt_010=float((diff > .1).float().mean()),
        probability_difference_gt_050=float((diff > .5).float().mean()),
    )
    return result


def tensor_fingerprint(tensors):
    digest = hashlib.sha256()
    dtypes = Counter()
    nonzero_b = total_b = 0
    finite = True
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
        dtypes[str(value.dtype)] += 1
        finite = finite and bool(torch.isfinite(value).all())
        if ".lora_B." in name:
            nonzero_b += int(torch.count_nonzero(value))
            total_b += value.numel()
    return {
        "sha256": digest.hexdigest(), "tensor_count": len(tensors),
        "dtypes": dict(dtypes), "all_finite": finite,
        "nonzero_b_elements": nonzero_b, "total_b_elements": total_b,
    }


def inspect_lora_slots(tensors, manager, scaling):
    """Compare every expected A/B matrix with its active vLLM GPU slot (TP=1)."""

    from vllm.lora.utils import parse_fine_tuned_lora_name

    ids = list(manager._active_adapters)
    if len(ids) != 1:
        raise RuntimeError(f"Expected one active LoRA, found {ids}")
    slot = manager.lora_index_to_id.index(ids[0])
    mapper = getattr(manager.model, "hf_to_vllm_mapper", None)
    expected = {}
    for name, value in tensors.items():
        module, is_a, is_bias = parse_fine_tuned_lora_name(name, mapper)
        if is_bias:
            raise ValueError("This diagnostic expects bias-free A/B adapters")
        expected[module, "A" if is_a else "B"] = value
    checks, seen = [], set()
    for name, module in manager.modules.items():
        children = manager.packed_modules.get(name, [name])
        for part, child in enumerate(children):
            for matrix, attribute in [("A", "lora_a_stacked"), ("B", "lora_b_stacked")]:
                key = child, matrix
                if key not in expected:
                    continue
                buffers = getattr(module, attribute)
                source = expected[key]
                target = buffers[part][slot, 0, :source.shape[0], :source.shape[1]].detach().cpu()
                converted = source.to(dtype=target.dtype)
                if matrix == "B":
                    converted = converted * scaling
                delta = (target.float() - converted.float()).abs()
                checks.append({
                    "module": child, "matrix": matrix, "shape": list(source.shape),
                    "gpu_slot_dtype": str(target.dtype),
                    "matches": bool(torch.equal(target, converted)),
                    "max_abs_difference": float(delta.max()) if torch.isfinite(delta).all() else None,
                })
                seen.add(key)
    missing = sorted(f"{name}.lora_{matrix}" for name, matrix in expected.keys() - seen)
    return {
        "lora_id": ids[0], "slot": slot, "scaling": scaling,
        "expected_matrices": len(expected), "checked_matrices": len(checks),
        "missing": missing,
        "all_match": not missing and all(item["matches"] for item in checks),
        "matrices": checks,
    }


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.terminal.write(value)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _git_revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _configure_process(seed):
    os.environ.update({
        "VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TOKENIZERS_PARALLELISM": "false", "DISABLE_WORKER_INIT": "1",
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
    })
    for name in ("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"):
        os.environ.pop(name, None)
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ["OMP_NUM_THREADS"]) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(seed)


def _actor_log_probs(worker, batch, *, remove_padding, micro_batch_size, adapter=True, temperature=1.0,
                     fp32_head=False, fp32_logprob=False, no_reduced_bf16=False, trace=None, calculate_entropy=False):
    from verl.protocol import pad_dataproto_to_divisor
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

    data = batch.select(
        batch_keys=["input_ids", "responses", "attention_mask", "position_ids"],
        non_tensor_batch_keys=["multi_modal_inputs"], deepcopy=True,
    )
    count = len(data)
    data, _ = pad_dataproto_to_divisor(data, micro_batch_size)
    data.meta_info.update(micro_batch_size=micro_batch_size, temperature=temperature, use_dynamic_bsz=False)
    previous = worker.actor.use_remove_padding
    worker.actor.use_remove_padding = remove_padding
    if worker._is_offload_param:
        load_fsdp_model_to_gpu(worker.actor_module_fsdp)
    try:
        context = nullcontext() if adapter else worker.actor.actor_module.disable_adapter()
        logprob_module = None
        if fp32_logprob:
            from verl.workers.actor import dp_actor
            logprob_module = dp_actor
        trace_context = (trace.attach(worker.actor_module_fsdp, data, micro_batch_size=micro_batch_size,
                                      remove_padding=remove_padding) if trace is not None else nullcontext())
        with worker.ulysses_sharding_manager, context, precision_control(
                worker.actor_module_fsdp, fp32_head=fp32_head, no_reduced_bf16=no_reduced_bf16,
                logprob_module=logprob_module), trace_context:
            with torch.no_grad():
                values, _ = worker.actor.compute_log_prob(data.to(torch.cuda.current_device()), calculate_entropy=calculate_entropy)
        return values[:count].detach().float().cpu()
    finally:
        worker.actor.use_remove_padding = previous
        if worker._is_offload_param:
            offload_fsdp_model_to_cpu(worker.actor_module_fsdp)
        torch.cuda.empty_cache()


def _evaluate_update_forwards(worker, batch, case, values, *, temperature, persist):
    from verl.protocol import pad_dataproto_to_divisor
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
    from adaptive_vision_rl.verl.update_diagnostics import update_mode_log_probs

    case["update_forward_evidence"] = {}
    for phase in ("actor_old_micro4", "actor_old_micro2", "actor_update_micro2", "actor_update_micro2_repeat"):
        print(f"  {phase}", flush=True)
        try:
            if phase.startswith("actor_old"):
                values[phase] = _actor_log_probs(
                    worker, batch, remove_padding=worker.config.model.use_remove_padding,
                    micro_batch_size=4 if phase.endswith("4") else 2,
                    temperature=temperature, calculate_entropy=True)
            else:
                data = batch.select(batch_keys=["input_ids", "responses", "attention_mask", "position_ids"],
                                    non_tensor_batch_keys=["multi_modal_inputs"], deepcopy=True)
                count = len(data)
                data, _ = pad_dataproto_to_divisor(data, 2)
                evidence = case["update_forward_evidence"].setdefault(phase, {})
                if worker._is_offload_param:
                    load_fsdp_model_to_gpu(worker.actor_module_fsdp)
                try:
                    with worker.ulysses_sharding_manager:
                        data = worker.ulysses_sharding_manager.preprocess_data(data.to(torch.cuda.current_device()))
                        micro_batches = ({**micro.batch, **micro.non_tensor_batch}
                                         for micro in data.chunk(len(data) // 2))
                        values[phase] = update_mode_log_probs(worker.actor, micro_batches,
                                                              temperature=temperature, evidence=evidence)[:count]
                finally:
                    if worker._is_offload_param:
                        offload_fsdp_model_to_cpu(worker.actor_module_fsdp)
        except Exception:
            case["errors"][phase] = traceback.format_exc()
            print(case["errors"][phase], file=sys.stderr, flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            persist()


def _evaluate_actor_variants(worker, batch, case, values, *, micro, remove_padding, temperature,
                             diagnostic_records, output, persist):
    enabled = diagnostic_records is not None
    focuses = focus_tokens(diagnostic_records) if enabled else []
    traces, references = {"focus_tokens": focuses, "phases": {}}, {}
    required_references = {"actor_training", "actor_base_training", "actor_fp32_head_training",
                           "actor_no_reduced_bf16_training", "actor_precise_head_training"}
    for phase, options in actor_variants(micro=micro, remove_padding=remove_padding,
                                         temperature=temperature, diagnostics=enabled):
        print(f"  {phase}", flush=True)
        try:
            trace = ForwardTrace(focuses) if enabled else None
            values[phase] = _actor_log_probs(worker, batch, trace=trace, **options)
            if trace is not None:
                if phase in required_references:
                    references[phase] = trace
                reference_name = trace_reference(phase)
                reference = references[reference_name]
                summary = trace.compare(reference)
                summary["reference_phase"] = reference_name
                summary["controls"] = options
                summary["focus_probabilities"] = []
                for focus in focuses:
                    probability = float(values[phase][focus["row"], focus["response_offset"]].exp())
                    summary["focus_probabilities"].append(
                        {**focus, "probability": probability if math.isfinite(probability) else None})
                traces["phases"][phase] = summary
                print(f"    captured {summary['capture_count']} activations; first changed captured stage: "
                      f"{summary['first_nonidentical_captured_stage']}; focus probabilities: "
                      f"{[round(item['probability'], 6) if item['probability'] is not None else None for item in summary['focus_probabilities']]}", flush=True)
        except Exception:
            case["errors"][phase] = traceback.format_exc()
            print(case["errors"][phase], file=sys.stderr, flush=True)
            torch.cuda.empty_cache()
        if enabled:
            filename = f"{case['name']}_forward_trace.json"
            case["forward_diagnostics"] = {
                "trace_file": filename, "focus_tokens": focuses,
                "all_captures_valid": bool(traces["phases"]) and not case["errors"] and all(
                    entry["all_finite"] and not entry["missing"] and not entry["unexpected"]
                    and all(item["probability"] is not None for item in entry["focus_probabilities"])
                    for entry in traces["phases"].values()),
                "first_nonidentical_captured_stages": {
                    name: entry["first_nonidentical_captured_stage"] for name, entry in traces["phases"].items()
                },
                "phases": {
                    name: {"reference_phase": entry["reference_phase"],
                           "focus_probabilities": entry["focus_probabilities"]}
                    for name, entry in traces["phases"].items()
                },
            }
            _write_json(output / filename, traces)
        persist()


def _teacher_forced_log_probs(worker, batch, raw_prompts, image_groups, *, adapter):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    width = batch.batch["responses"].shape[1]
    values = torch.full((len(batch), width), float("nan"))
    alignments = []
    engine = worker.rollout.inference_engine
    with worker.rollout_sharding_manager:
        ids = list(engine.llm_engine.list_loras())
        if adapter and len(ids) != 1:
            raise RuntimeError(f"Expected one registered adapter, found {ids}")
        request = LoRARequest(str(ids[0]), ids[0], "/consistency-memory-adapter") if adapter else None
        for index in range(len(batch)):
            # Serial prefill bounds the full-vocabulary prompt-logprob memory.
            # Reset after sleep too: freed KV storage must never be reused here.
            engine.reset_prefix_cache()
            response_mask = batch.batch["attention_mask"][index, -width:].bool()
            response = batch.batch["responses"][index][response_mask].tolist()
            prompt_mask = batch.batch["attention_mask"][index, :-width].bool()
            prompt = batch.batch["prompts"][index][prompt_mask].tolist()
            expected = prompt + response
            output = engine.generate(
                prompts=[{"prompt_token_ids": list(raw_prompts[index]) + response,
                          "multi_modal_data": {"image": image_groups[index]}}],
                sampling_params=SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1,
                                               prompt_logprobs=0, logprobs=0, detokenize=False),
                lora_request=request, use_tqdm=False,
            )[0]
            row_values, alignment = response_prompt_log_probs(output, expected, len(prompt), len(response))
            values[index, :len(response)] = row_values
            alignments.append(alignment)
    return values, alignments


def _reference_actions(rows):
    actions = []
    for row in rows:
        boxes = row["env_kwargs"].get("reference_boxes") or []
        box = boxes[0] if len(boxes) else [.25, .25, .75, .75]
        # The tool protocol uses 0..1000 coordinates, independent of PIL size.
        x0 = max(0, min(999, math.floor(float(box[0]) * 1000)))
        y0 = max(0, min(999, math.floor(float(box[1]) * 1000)))
        x1 = max(x0 + 1, min(1000, math.ceil(float(box[2]) * 1000)))
        y1 = max(y0 + 1, min(1000, math.ceil(float(box[3]) * 1000)))
        payload = {"name": "request_local_region", "arguments": {"bbox_2d": [x0, y0, x1, y1]}}
        actions.append("<think>Inspect the selected region.</think><tool_call>" + json.dumps(payload) + "</tool_call>")
    return actions


def _make_batch(worker, collector, rows, observation, replay_records=None):
    from verl import DataProto

    dummy = DataProto.from_dict(
        tensors={"input_ids": torch.zeros((len(rows), 1), dtype=torch.long)},
        non_tensors={"raw_prompt": object_array([row.get("prompt") for row in rows]),
                     "data_source": object_array([row.get("data_source", "consistency") for row in rows])},
    )
    batch = collector.preprocess_batch(gen_batch=dummy, obs=observation)
    raw_prompts = [list(item) for item in batch.non_tensor_batch["raw_prompt_ids"]]
    images = [item["image"] for item in batch.non_tensor_batch["multi_modal_data"]]
    non_tensor_keys = ["raw_prompt_ids", "multi_modal_data"]
    if "raw_prompt" in batch.non_tensor_batch:
        non_tensor_keys.append("raw_prompt")
    generation = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=non_tensor_keys,
    )
    if replay_records is None:
        # Exactly the normal rollout worker method, including sharding-manager sync.
        output = worker.generate_sequences(generation)
    else:
        for index, record in enumerate(replay_records):
            if str(rows[index]["env_kwargs"]["sample_id"]) != record["anchor"]["sample_id"]:
                raise ValueError(f"Replay sample order changed at row {index}")
            if [list(image.size) for image in images[index]] != record["image_sizes"]:
                raise ValueError(f"Replay image sizes changed at row {index}")
            if batch.non_tensor_batch["multi_modal_inputs"][index]["image_grid_thw"].tolist() != record["image_grid_thw"]:
                raise ValueError(f"Replay image grids changed at row {index}")
        output = DataProto.from_dict(tensors=replay_tensors(
            generation.batch, replay_records, response_width=worker.config.rollout.response_length,
            pad_token_id=worker.tokenizer.pad_token_id,
        ))
    batch = batch.union(output)
    return batch, raw_prompts, images


def _sample_records(batch, images, generation_outputs, tokenizer, values, top_k):
    width = batch.batch["responses"].shape[1]
    rows = []
    for index in range(len(batch)):
        mask = batch.batch["attention_mask"][index, -width:].bool()
        prompt_mask = batch.batch["attention_mask"][index, :-width].bool()
        response = batch.batch["responses"][index][mask].tolist()
        prompt = batch.batch["prompts"][index][prompt_mask].tolist()
        row_values = {name: tensor[index][mask].tolist() for name, tensor in values.items()}
        row_values = {name: [v if math.isfinite(v) else None for v in seq] for name, seq in row_values.items()}
        top = []
        comparison_phase = "actor_old_micro4" if "actor_old_micro4" in values else "actor_training"
        if comparison_phase in values:
            left, right = values["rollout"][index], values[comparison_phase][index]
            finite = mask & torch.isfinite(left) & torch.isfinite(right)
            diff = (left.exp() - right.exp()).abs().masked_fill(~finite, -1)
            for offset in diff.topk(min(top_k, int(finite.sum()))).indices.tolist():
                top.append({
                    "response_offset": offset, "token_id": int(batch.batch["responses"][index, offset]),
                    "token_text": tokenizer.decode([int(batch.batch["responses"][index, offset])]),
                    "context": tokenizer.decode(response[max(0, offset - 12):offset + 13]),
                    "probability_difference": float(diff[offset]),
                    "log_probs": {name: float(value[index, offset]) if torch.isfinite(value[index, offset]) else None
                                  for name, value in values.items()},
                })
        positions = batch.batch["position_ids"][index]
        valid_positions = batch.batch["attention_mask"][index].bool()
        rows.append({
            "row": index, "anchor": dict(batch.non_tensor_batch["anchor_obs"][index]),
            "image_sizes": [list(image.size) for image in images[index]],
            "image_grid_thw": batch.non_tensor_batch["multi_modal_inputs"][index]["image_grid_thw"].tolist(),
            "prompt_token_ids": prompt, "response_token_ids": response,
            "position_ids": positions[..., valid_positions].tolist(),
            "generation_prompt_alignment": token_alignment(prompt, generation_outputs[index].prompt_token_ids),
            "response_text": tokenizer.decode(response, skip_special_tokens=False),
            "log_probs": row_values, "largest_differences": top,
        })
    return rows


def _run_gpu(args, report, persist):
    if not torch.cuda.is_available():
        raise RuntimeError("This verification requires the Linux CUDA training environment")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run with python on one GPU; do not use torchrun")
    _configure_process(args.seed)
    import verl
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.utils import hf_processor
    from adaptive_vision_rl.environment import AdaptiveVisionEnvironmentManager
    from adaptive_vision_rl.thinking_template import configure_thinking_tokenizer
    from adaptive_vision_rl.verl.collector import AdaptiveVisionTrajectoryCollector
    from adaptive_vision_rl.verl.lora_checkpoint import select_lora_tensors
    from adaptive_vision_rl.verl.lora_sync import collect_lora_params

    project = Path(__file__).resolve().parents[2]
    expected_revision = (project / "third_party/verl-agent.commit").read_text().strip()
    actual_revision = _git_revision(Path(verl.__file__).resolve().parents[1])
    report["environment"].update(
        verl_commit=actual_revision, expected_verl_commit=expected_revision,
        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
    )
    if actual_revision != expected_revision and not args.allow_version_mismatch:
        raise ValueError("verl checkout does not match third_party/verl-agent.commit; use the training checkout or explicitly pass --allow-version-mismatch")
    replay_source, replay_records = None, {}
    if getattr(args, "replay_report", None):
        source_path = args.replay_report.expanduser().resolve()
        if source_path.is_dir():
            source_path = source_path / "report.json"
        replay_source = json.loads(source_path.read_text())
        if replay_source["status"] != "completed" or not replay_source["checkpoint_lora_matches_actor"]:
            raise ValueError("Replay requires a completed source report with matching checkpoint weights")
        config = OmegaConf.create(replay_source["effective_config"])
        for case in replay_source["cases"]:
            name = case["name"]
            replay_records[name] = json.loads((source_path.parent / f"{name}_samples.json").read_text())
        if "decision" not in replay_records:
            raise ValueError("Replay source is missing the decision scenario")
        report["replay"] = {"source_report": str(source_path), "source_report_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                            "source_project_commit": replay_source.get("project_commit"),
                            "rollout_probabilities": "Reused from source report; responses are not resampled",
                            "generation_alignment": "Reconstructed inputs are checked against the original generation inputs"}
    else:
        config = OmegaConf.merge(
            OmegaConf.load(Path(verl.__file__).resolve().parent / "trainer/config/ppo_trainer.yaml"),
            OmegaConf.load(args.config), OmegaConf.from_dotlist(args.override),
        )
    OmegaConf.resolve(config)
    report["checkpoint_lora_config_source"] = (
        "Replay source effective_config; .pt does not encode LoRA alpha" if replay_source else
        "config YAML plus CLI overrides; .pt does not encode LoRA alpha")
    rollout = config.actor_rollout_ref.rollout
    update_probe = bool(getattr(args, "update_forward_diagnostics", False))
    if update_probe and (int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu) != 2
                         or config.actor_rollout_ref.actor.entropy_coeff != 0
                         or config.actor_rollout_ref.actor.use_dynamic_bsz
                         or int(config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)) != 1):
        raise ValueError("Update probe requires actor micro=2, entropy_coeff=0, fixed batches and SP=1")
    if (rollout.name != "vllm" or rollout.mode != "sync" or rollout.tensor_model_parallel_size != 1
            or rollout.n != 1 or config.actor_rollout_ref.actor.strategy != "fsdp"
            or config.actor_rollout_ref.model.lora_rank <= 0):
        raise ValueError("This diagnostic requires sync vLLM, TP=1, rollout.n=1, FSDP1 and LoRA")
    if config.actor_rollout_ref.model.get("use_fused_kernels", False):
        raise ValueError("This diagnostic currently requires use_fused_kernels=false")
    if args.base_model:
        config.actor_rollout_ref.model.path = args.base_model
    if args.data_root:
        config.env.adaptive_vision.data_root = str(args.data_root)
    else:
        data_root = Path(config.env.adaptive_vision.data_root).expanduser()
        if not data_root.is_absolute():
            config.env.adaptive_vision.data_root = str(project / data_root)
    if args.max_response_tokens is not None:
        config.data.max_response_length = args.max_response_tokens
        rollout.response_length = args.max_response_tokens
    # Teacher forcing adds one unused generated token to the full sequence.
    rollout.max_model_len = max(int(rollout.get("max_model_len") or 0),
                                int(config.data.max_prompt_length + config.data.max_response_length + 1))
    config.actor_rollout_ref.actor.optim.total_training_steps = max(1, int(config.actor_rollout_ref.actor.optim.get("total_training_steps") or 0))
    report["effective_config"] = OmegaConf.to_container(config, resolve=True)
    (args.output / "effective_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    all_rows = pq.read_table(args.parquet).to_pylist()
    if replay_source is not None:
        selected = replay_source["selected_samples"]
        matches = {sample: [row for row in all_rows if str(row["env_kwargs"]["sample_id"]) == sample] for sample in selected}
        if any(len(rows) != 1 for rows in matches.values()):
            raise ValueError("Source report sample IDs are missing or duplicated in the parquet")
        rows = [matches[sample][0] for sample in selected]
    else:
        rows = all_rows[args.offset:args.offset + args.samples]
        if len(rows) != args.samples:
            raise ValueError("Selected parquet slice contains fewer rows than --samples")
    report["selected_samples"] = [str(row["env_kwargs"]["sample_id"]) for row in rows]
    persist()

    print("Initializing the original FSDP actor and sync vLLM worker...", flush=True)
    worker = ActorRolloutRefWorker(config.actor_rollout_ref, role="actor_rollout")
    worker.init_model()
    configure_thinking_tokenizer(worker.tokenizer)
    processor = hf_processor(config.actor_rollout_ref.model.path, use_fast=True,
                             trust_remote_code=config.data.trust_remote_code)
    configure_thinking_tokenizer(processor.tokenizer)
    checkpoint_file = args.checkpoint / "actor/model_world_size_1_rank_0.pt"
    print(f"Loading model weights only from {checkpoint_file}", flush=True)
    state = torch.load(checkpoint_file, map_location="cpu", mmap=True, weights_only=False)
    source_lora = select_lora_tensors(state, rank=int(config.actor_rollout_ref.model.lora_rank))
    report["checkpoint_lora"] = tensor_fingerprint(source_lora)
    with FSDP.state_dict_type(worker.actor_module_fsdp, StateDictType.SHARDED_STATE_DICT,
                              ShardedStateDictConfig(offload_to_cpu=True)):
        incompatible = worker.actor_module_fsdp.load_state_dict(state, strict=True)
    report["checkpoint_load"] = {"missing_keys": list(incompatible.missing_keys), "unexpected_keys": list(incompatible.unexpected_keys)}
    del source_lora, state
    gc.collect()
    live = collect_lora_params(worker.actor_module_fsdp)
    report["loaded_actor_lora"] = tensor_fingerprint(live)
    report["checkpoint_lora_matches_actor"] = report["checkpoint_lora"] == report["loaded_actor_lora"]
    if not report["checkpoint_lora_matches_actor"]:
        raise RuntimeError("The loaded actor LoRA does not match the checkpoint")
    if replay_source is not None:
        report["replay"]["actor_lora_matches_source"] = report["loaded_actor_lora"] == replay_source["loaded_actor_lora"]
        if not report["replay"]["actor_lora_matches_source"]:
            raise ValueError("Current checkpoint LoRA differs from the replay source")
    del live
    report["sync_events"] = []
    manager = worker.rollout_sharding_manager
    original_update = manager.update_params

    def inspect_update(updated_params, peft_config=None):
        original_update(updated_params, peft_config=peft_config)
        event = {"event": len(report["sync_events"]), "input_tensor_count": len(updated_params)}
        report["sync_events"].append(event)
        try:
            tensors = collect_lora_params(worker.actor_module_fsdp)
            event["actor_lora"] = tensor_fingerprint(tensors)
            lora_manager = manager.model_runner.lora_manager._adapter_manager
            factor = peft_config.lora_alpha / (math.sqrt(peft_config.r) if peft_config.use_rslora else peft_config.r)
            event["gpu_slots"] = inspect_lora_slots(tensors, lora_manager, factor)
        except Exception:
            # Inspection failure must be reported without losing probability data.
            event["inspection_error"] = traceback.format_exc()
        persist()

    manager.update_params = inspect_update
    engine = worker.rollout.inference_engine
    vllm_config = engine.llm_engine.vllm_config
    report["runtime"] = {
        "actor_master_dtype": str(next(worker.actor_module_fsdp.parameters()).dtype),
        "actor_mixed_precision": str(worker.actor_module_fsdp.mixed_precision),
        "actor_text_attention": worker.actor_model_config.text_config._attn_implementation,
        "vllm_dtype": str(vllm_config.model_config.dtype),
        "vllm_logprobs_mode": getattr(vllm_config.model_config, "logprobs_mode", None),
        "vllm_lora_dtype": str(vllm_config.lora_config.lora_dtype),
        "vllm_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "flags": {key: os.environ.get(key) for key in ["VLLM_USE_V1", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_ATTENTION_BACKEND", "PYTORCH_CUDA_ALLOC_CONF"]},
    }
    original_generate = engine.generate
    captured = []
    last_request = {}

    def capture_generate(*positional, **kwargs):
        request = kwargs.get("lora_request")
        requests = request if isinstance(request, (list, tuple)) else [request]
        last_request["lora_ids"] = sorted({item.lora_int_id for item in requests if item is not None})
        sampling = kwargs.get("sampling_params")
        last_request["sampling"] = {key: getattr(sampling, key, None) for key in ["temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "logprobs", "prompt_logprobs"]}
        outputs = original_generate(*positional, **kwargs)
        captured[:] = outputs
        return outputs

    engine.generate = capture_generate
    collector = AdaptiveVisionTrajectoryCollector(config=config, tokenizer=worker.tokenizer, processor=processor)
    environment = AdaptiveVisionEnvironmentManager(config, processor, is_train=False)
    initial_observation, _ = environment.reset([row["env_kwargs"] for row in rows])
    observations = [("decision", initial_observation)]
    if not args.skip_reference_second_turn and (replay_source is None or "reference_second_turn" in replay_records):
        observation, _, done, _ = environment.step(_reference_actions(rows))
        if np.asarray(done).any():
            raise RuntimeError("Controlled tool requests did not produce answer-after-tool observations")
        observations.append(("reference_second_turn", observation))

    mixed_sources = []
    for name, observation in observations:
        print(f"Comparing {name}: {len(rows)} samples...", flush=True)
        case = {"name": name, "errors": {}, "comparisons": {}}
        report["cases"].append(case)
        saved = replay_records.get(name)
        batch, raw_prompts, images = _make_batch(worker, collector, rows, observation, replay_records=saved)
        if saved is None:
            generation_outputs = list(captured)
            case["generation_lora_ids"] = last_request["lora_ids"]
            case["generation_sampling"] = last_request["sampling"]
            case["generation_sync_event"] = len(report["sync_events"]) - 1
        else:
            generation_outputs = [SimpleNamespace(prompt_token_ids=record["prompt_token_ids"]) for record in saved]
            case["generation_replayed"] = True
            case["generation_prompt_alignment_source"] = "Reconstructed prompt vs saved original generation prompt"
        values = {"rollout": batch.batch["rollout_log_probs"].float().cpu()}
        temperature = float(rollout.temperature)
        micro = int(rollout.log_prob_micro_batch_size_per_gpu)
        original_rmpad = bool(config.actor_rollout_ref.model.use_remove_padding)
        if update_probe:
            _evaluate_update_forwards(worker, batch, case, values, temperature=temperature, persist=persist)
        else:
            _evaluate_actor_variants(worker, batch, case, values, micro=micro, remove_padding=original_rmpad,
                                 temperature=temperature, diagnostic_records=saved if args.forward_diagnostics else None,
                                 output=args.output, persist=persist)
        for phase, enabled in ([] if update_probe else [("vllm_prefill", True), ("vllm_base_prefill", False)]):
            print(f"  {phase}", flush=True)
            try:
                values[phase], case[phase + "_alignment"] = _teacher_forced_log_probs(worker, batch, raw_prompts, images, adapter=enabled)
            except Exception:
                case["errors"][phase] = traceback.format_exc()
                torch.cuda.empty_cache()
            persist()
        mask = batch.batch["attention_mask"][:, -batch.batch["responses"].shape[1]:].bool().cpu()
        comparisons = [
            ("rollout", "actor_training"), ("actor_training", "actor_packed_single"),
            ("actor_packed_single", "actor_padded_single"), ("rollout", "vllm_prefill"),
            ("vllm_prefill", "actor_raw" if temperature != 1.0 else "actor_training"),
            ("vllm_base_prefill", "actor_base"),
            ("vllm_prefill", "vllm_base_prefill"),
            ("actor_raw" if temperature != 1.0 else "actor_padded_single", "actor_base"),
            ("rollout", "actor_base"),
        ]
        if args.forward_diagnostics:
            comparisons.extend(extra_comparisons(values, temperature=temperature))
        for left, right in comparisons:
            if left in values and right in values:
                case["comparisons"][f"{left}_vs_{right}"] = probability_difference(values[left], values[right], mask)
        if update_probe:
            _record_update_comparisons(case, values, mask, config.actor_rollout_ref.actor)
        case["rows"] = len(batch)
        case["response_tokens"] = int(mask.sum())
        case["sampling_temperature"] = temperature
        case["actor_micro_batch_size"] = micro
        case["actor_remove_padding"] = original_rmpad
        records = _sample_records(batch, images, generation_outputs, worker.tokenizer, values, args.top_tokens)
        case["generation_prompt_alignment_all_match"] = all(item["generation_prompt_alignment"]["matches"] for item in records)
        _write_json(args.output / f"{name}_samples.json", records)
        persist()
        mixed_sources.append((batch, images, generation_outputs, values))
        del records
        gc.collect()
        torch.cuda.empty_cache()

    if len(mixed_sources) == 2:
        from verl import DataProto

        print("Comparing interleaved one-image/two-image actor micro-batches...", flush=True)
        case = {"name": "mixed_turns", "errors": {}, "comparisons": {}}
        report["cases"].append(case)
        order = np.arange(2 * len(rows)).reshape(2, len(rows)).T.reshape(-1)
        batch = DataProto.concat([source[0] for source in mixed_sources]).select_idxs(order)
        combined_images = sum([source[1] for source in mixed_sources], [])
        combined_outputs = sum([source[2] for source in mixed_sources], [])
        images = [combined_images[i] for i in order]
        generation_outputs = [combined_outputs[i] for i in order]
        shared = set.intersection(*(set(source[3]) for source in mixed_sources))
        values = {name: torch.cat([source[3][name] for source in mixed_sources])[order] for name in shared}
        if "actor_training" in values:
            values["actor_separate_turns"] = values.pop("actor_training")
        if update_probe:
            values = {name: value for name, value in values.items() if not name.startswith("actor_")}
            _evaluate_update_forwards(worker, batch, case, values, temperature=temperature, persist=persist)
        elif args.forward_diagnostics:
            # Every actor control must actually run with the mixed grouping.
            # Keep only the explicitly named separate-turn reference and vLLM data.
            values = {name: value for name, value in values.items()
                      if not name.startswith("actor_") or name == "actor_separate_turns"}
            diagnostic_records = replay_records.get("mixed_turns")
            if diagnostic_records is None:
                source_records = replay_records["decision"] + replay_records["reference_second_turn"]
                diagnostic_records = [source_records[i] for i in order]
            width = batch.batch["responses"].shape[-1]
            reconstructed = replay_tensors(
                {name: batch.batch[name][..., :-width] for name in ("input_ids", "attention_mask", "position_ids")},
                diagnostic_records, response_width=width, pad_token_id=worker.tokenizer.pad_token_id,
            )
            if not torch.equal(reconstructed["responses"], batch.batch["responses"]):
                raise ValueError("Mixed replay responses differ from the source report")
            _evaluate_actor_variants(worker, batch, case, values, micro=micro, remove_padding=original_rmpad,
                                     temperature=temperature, diagnostic_records=diagnostic_records,
                                     output=args.output, persist=persist)
        else:
            try:
                values["actor_training"] = _actor_log_probs(worker, batch, remove_padding=original_rmpad,
                                                           micro_batch_size=micro, temperature=temperature)
            except Exception:
                case["errors"]["actor_training"] = traceback.format_exc()
                torch.cuda.empty_cache()
        mask = batch.batch["attention_mask"][:, -batch.batch["responses"].shape[1]:].bool().cpu()
        comparisons = [("rollout", "actor_training"), ("actor_training", "actor_separate_turns"),
                       ("actor_training", "actor_packed_single"), ("actor_training", "actor_padded_single")]
        if args.forward_diagnostics:
            comparisons.extend(extra_comparisons(values, temperature=temperature))
        for left, right in comparisons:
            if left in values and right in values:
                case["comparisons"][f"{left}_vs_{right}"] = probability_difference(values[left], values[right], mask)
        if update_probe:
            _record_update_comparisons(case, values, mask, config.actor_rollout_ref.actor)
        case.update(rows=len(batch), response_tokens=int(mask.sum()), sampling_temperature=temperature,
                    actor_micro_batch_size=micro, actor_remove_padding=original_rmpad,
                    note="Reuses the exact responses above; only actor row grouping changes, interleaving one-image/two-image rows.")
        if replay_source is not None:
            case["generation_replayed"] = True
        records = _sample_records(batch, images, generation_outputs, worker.tokenizer, values, args.top_tokens)
        case["generation_prompt_alignment_all_match"] = all(item["generation_prompt_alignment"]["matches"] for item in records)
        _write_json(args.output / "mixed_turns_samples.json", records)
        persist()

    final_lora = tensor_fingerprint(collect_lora_params(worker.actor_module_fsdp))
    report["final_actor_lora"] = final_lora
    report["actor_weights_unchanged"] = final_lora == report["loaded_actor_lora"]
    report["precision_controls_restored"] = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        == report["runtime"]["cuda_matmul_allow_bf16_reduced_precision_reduction"])
    incomplete = (not report["actor_weights_unchanged"] or not report["precision_controls_restored"] or any(
                      case["errors"] or any(value["nonfinite_tokens"] for value in case["comparisons"].values())
                      or any(value["nonfinite_tokens"] for value in case.get("update_ratios", {}).values())
                      or not case.get("forward_diagnostics", {}).get("all_captures_valid", True)
                      for case in report["cases"])
                  or any("inspection_error" in event for event in report["sync_events"]))
    report["status"] = "partial" if incomplete else "completed"


def _record_update_comparisons(case, values, mask, actor_config):
    from adaptive_vision_rl.verl.update_diagnostics import update_comparisons
    low = actor_config.clip_ratio_low
    high = actor_config.clip_ratio_high
    case["comparisons"], case["update_ratios"] = update_comparisons(
        values, mask, clip_low=float(actor_config.clip_ratio if low is None else low),
        clip_high=float(actor_config.clip_ratio if high is None else high))


def _markdown_report(report):
    if report.get("update_forward_diagnostics"):
        return _update_markdown_report(report)
    lines = ["# 训推一致性验证报告", "", f"状态：`{report['status']}`", "",
             f"Checkpoint：`{report['checkpoint']}`", "",
             "该脚本不更新参数，不保存或删除 checkpoint；读取完整 .pt 中的模型权重，不依赖独立 adapter 文件。", "",
             f"Checkpoint LoRA 与 actor 完全相同：`{report.get('checkpoint_lora_matches_actor', '未完成')}`。",
             f"检查前后 actor LoRA 未改变：`{report.get('actor_weights_unchanged', '未完成')}`。",
             f"源 checkpoint 文件大小与修改时间未改变：`{report.get('checkpoint_file_unchanged', '未完成')}`。", "",
             "`mean/std/max` 为同一批 response token 的概率绝对差；std 使用样本标准差，与训练指标一致。", "",
             "| 场景 | 对照 | mean | std | max | 缺失/非有限 token |", "|---|---|---:|---:|---:|---:|"]
    if report.get("replay"):
        lines[4:4] = [
            f"固定回答重放，来源：`{report['replay']['source_report']}`。", "",
            "`rollout` 概率沿用原报告；本轮没有重新生成回答。actor 与 vllm_prefill 使用当前代码重算。",
            "重建输入时核对原 prompt token、位置编码、图像尺寸/grid 和 response；LoRA 指纹必须与原报告相同。", "",
        ]
    for case in report["cases"]:
        for name, item in case["comparisons"].items():
            numbers = [f"{item[key]:.6f}" if key in item else "n/a" for key in ("mean", "std", "max")]
            lines.append(f"| {case['name']} | {name} | {' | '.join(numbers)} | {item['nonfinite_tokens']} |")
    for case in report["cases"]:
        diagnostic = case.get("forward_diagnostics")
        if not diagnostic:
            continue
        focuses = diagnostic["focus_tokens"]
        lines.extend(["", f"## {case['name']} 固定异常 token 对照", "",
                      f"激活记录有效：`{diagnostic['all_captures_valid']}`。详见 `{diagnostic['trace_file']}`。", "",
                      "| 前向阶段 | 激活比较基准 | 最早出现变化的已记录阶段 | "
                      + " | ".join(f"row {item['row']} / response {item['response_offset']} / ID {item['token_id']} 概率" for item in focuses) + " |",
                      "|---|---|---|" + "---:|" * len(focuses)])
        for phase, entry in diagnostic["phases"].items():
            first = diagnostic["first_nonidentical_captured_stages"][phase] or "记录值完全相同"
            probabilities = " | ".join(f"{item['probability']:.6f}" if item["probability"] is not None else "n/a"
                                       for item in entry["focus_probabilities"])
            lines.append(f"| {phase} | {entry['reference_phase']} | {first} | {probabilities} |")
        lines.extend(["", "激活只记录选中 token 的预测位置，以及其所在样本的视觉 merger/DeepStack 输出。",
                      "最早变化阶段仅限这些记录点，不代表已证明该模块是根因；没有记录其他位置或模块内部的运算。"])
    if report.get("forward_diagnostics"):
        lines.extend(["", "重复前向及控制后的复算检查稳定性；base 系列均关闭 LoRA并使用温度 1。",
                      "fp32_logprob 只将 logprob 输入转为 FP32；fp32_head 临时用 FP32 计算输出层，其他层保持原精度。",
                      "no_reduced_bf16 临时禁止 BF16 GEMM 的低精度 reduction；precise_head 同时启用输出层与 reduction 控制。",
                      "这些控制仅用于定位，没有写入正式训练配置。概率差改善本身不能证明某个内核或模块实现有错。",
                      f"临时 BF16 reduction 设置已恢复：`{report.get('precision_controls_restored', '未完成')}`。"])
    sync_label = "同步编号（重放时首次为 vLLM 固定 token 重算前）" if report.get("replay") else "同步次数（0 为首次采样前）"
    lines.extend(["", "## LoRA 同步检查", "",
                  f"| {sync_label} | 已核对/应核对矩阵 | GPU 槽位与 actor 一致 |", "|---|---:|---|"])
    for event in report.get("sync_events", []):
        slots = event.get("gpu_slots", {})
        lines.append(f"| {event['event']} | {slots.get('checked_matrices', '?')}/{slots.get('expected_matrices', '?')} | {slots.get('all_match', '检查失败，见 report.json')} |")
    errors = [f"{case['name']}/{phase}" for case in report["cases"] for phase in case["errors"]]
    if errors:
        lines.extend(["", "未完成的对照：" + "、".join(errors) + "。错误堆栈见 report.json。"])
    lines.extend(["", "## 如何定位", "",
                  "- 先看 checkpoint 与 actor 权重是否相同，以及 sync_events 中 GPU LoRA 槽位是否 all_match。", 
                  "- generation_prompt_alignment 或 prefill_alignment 不匹配：先查图像展开、token 输入和 response 对齐。",
                  "- actor_training 与 actor_packed_single 差异明显：关注 batching；packed_single 与 padded_single 差异明显：关注去 padding。",
                  "- mixed_turns 的 actor_training 与 actor_separate_turns 差异明显：关注单图/双图混合 micro-batch 的输入拼接和样本隔离。",
                  "- rollout 与 vllm_prefill 差异明显：关注 vLLM decode/prefill、KV 缓存和 LoRA 内核路径。",
                  "- base 对照接近但启用 LoRA 后明显分歧：优先查 LoRA 的计算；base 对照也分歧：优先查共同的模型前向和多模态输入。",
                  "- vllm_prefill 与 vllm_base_prefill 检查 adapter 对 vLLM 的实际影响；结合 actor 的 LoRA/base 对照，可识别权重已加载但计算未生效的情况。",
                  "- temperature 非 1 时，vLLM 默认 raw logprobs 与 actor_training 的温度缩放口径不同；使用 actor_raw 对照。",
                  "", "## 文件", "", "- report.json：完整配置、版本、汇总、同步权重检查与失败堆栈。",
                  "- *_samples.json：输入 token、位置编码、逐 token logprob、输出文本及最大差异 token。",
                  "- *_forward_trace.json（深度诊断模式）：重复/关闭 LoRA/精度对照下的指定位置激活差、logits 和概率。",
                  "- effective_config.yaml / run.log：复现配置与运行日志。", "",
                  "reference_second_turn 由参考框（缺失时用中心框）构造固定双图观察，专门检查第二轮；不是模型自然选择工具的比例。",
                  "各对照共享同一段已经采样的 response。没有把不同生成结果的概率直接相减。", "",
                  "LoRA rank、alpha、基座路径来自配置；.pt 无法自动提供 alpha，必须使用原训练值。",
                  "completed 表示所有诊断阶段执行完毕，并不表示两端概率一致。", "",
                  "缺失阶段和非有限 token 不代表通过。报告保留诊断数据，由人工结合各对照判断原因，不使用任意阈值宣布修复。", ""])
    if report.get("fatal_error"):
        lines.extend(["## 运行失败", "", "```text", report["fatal_error"], "```", ""])
    return "\n".join(lines)


def _update_markdown_report(report):
    lines = ["# micro-batch=2 真实更新模式前向验证", "", f"状态：`{report['status']}`", "",
             f"Checkpoint：`{report['checkpoint']}`", "",
             "固定原报告的输入和回答；rollout 概率沿用原报告。本轮不重新采样、不运行 vLLM prefill、不改变计算精度。",
             "旧概率通过真实 compute_log_prob（eval/no_grad，含 entropy 计算）分别按 4/2 条重算。",
             "更新概率直接调用 PPO 的 _forward_micro_batch：train、启用梯度、启用 gradient checkpointing，每批 2 条。",
             "仅前向，不调用 backward 或 optimizer；独立短训才执行真实参数更新。", "",
             f"Checkpoint 与 actor LoRA 一致：`{report.get('checkpoint_lora_matches_actor', '未完成')}`。",
             f"前向前后 actor LoRA 未改变：`{report.get('actor_weights_unchanged', '未完成')}`。",
             f"源 checkpoint 文件未改变：`{report.get('checkpoint_file_unchanged', '未完成')}`。", "",
             "ratio = exp(后者 logprob − 前者 logprob)，在任何参数更新之前计算。",
             "超界比例按配置的 PPO clip 边界统计；它不是 pg_clipfrac，后者还依赖 advantage 的符号。", "",
             "| 场景 | 对照 | ratio min | ratio max | 超界比例 | 概率差 mean | 概率差 max | 非有限 token |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        for name, ratio in case.get("update_ratios", {}).items():
            diff = case["comparisons"][name]
            numbers = [f"{ratio[key]:.6f}" if key in ratio else "n/a" for key in ("min", "max", "fraction_outside_bounds")]
            differences = [f"{diff[key]:.6f}" if key in diff else "n/a" for key in ("mean", "max")]
            lines.append(f"| {case['name']} | {name} | {' | '.join(numbers + differences)} | {ratio['nonfinite_tokens']} |")
        for phase, evidence in case.get("update_forward_evidence", {}).items():
            lines.extend(["", f"{case['name']}/{phase}：真实模式证据有效 `{evidence.get('valid', False)}`；"
                          f"模式已恢复 `{evidence.get('training_modes_restored', False)}`。"])
        for phase, error in case["errors"].items():
            lines.extend(["", f"失败：{case['name']}/{phase}", "```text", error, "```"])
    lines.extend(["", "重点比较 old_micro4→update_micro2 与 old_micro2→update_micro2：后者是否使无更新时的 ratio 更接近 1、减少超界。",
                  "update_micro2 的重复对照检查本次前向稳定性。mixed_turns 混合单图/双图，复用相同 token，不能与另两场景相加当成独立样本。",
                  "completed 只表示检查执行完毕，不表示差异可接受或长期训练稳定。",
                  "逐 token 数据见 *_samples.json；模式证据、完整 ratio 分布、概率差、配置和版本见 report.json。", ""])
    if report.get("fatal_error"):
        lines.extend(["```text", report["fatal_error"], "```", ""])
    return "\n".join(lines)


def run_verification(args):
    args.output = args.output.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.output == args.checkpoint or args.checkpoint in args.output.parents:
        raise ValueError("Output must be outside the source checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("torch", "transformers", "peft", "vllm", "flash-attn", "ray", "tensordict"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "status": "running", "checkpoint": str(args.checkpoint), "cases": [],
        "command": sys.argv, "environment": {"python": sys.version, "platform": platform.platform(), "packages": versions},
        "seed": args.seed, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "read_only": True, "model_weights_only": True,
        "forward_diagnostics": bool(getattr(args, "forward_diagnostics", False)),
        "update_forward_diagnostics": bool(getattr(args, "update_forward_diagnostics", False)),
        "project_commit": _git_revision(Path(__file__).resolve().parents[2]),
    }
    checkpoint_file = args.checkpoint / "actor/model_world_size_1_rank_0.pt"
    before = None

    def persist():
        _write_json(args.output / "report.json", report)

    failed = False
    started = time.monotonic()
    with (args.output / "run.log").open("w", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                stat = checkpoint_file.stat()
                before = (stat.st_size, stat.st_mtime_ns)
                report["checkpoint_model_file"] = {"path": str(checkpoint_file), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                persist()
                _run_gpu(args, report, persist)
            except Exception:
                failed = True
                report["status"] = "failed"
                report["fatal_error"] = traceback.format_exc()
                print(report["fatal_error"], file=sys.stderr, flush=True)
            finally:
                report["elapsed_seconds"] = time.monotonic() - started
                if before is not None:
                    try:
                        stat = checkpoint_file.stat()
                        report["checkpoint_file_unchanged"] = before == (stat.st_size, stat.st_mtime_ns)
                    except OSError:
                        report["checkpoint_file_unchanged"] = False
                persist()
                (args.output / "REPORT.md").write_text(_markdown_report(report))
    archive = args.output.with_name(args.output.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(args.output.iterdir()):
            if path.is_file():
                bundle.write(path, f"{args.output.name}/{path.name}")
    print(f"Report: {args.output / 'REPORT.md'}\nDownload this archive: {archive}", flush=True)
    return 1 if failed or report["status"] == "partial" else 0
