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

import numpy as np
import torch


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


def _actor_log_probs(worker, batch, *, remove_padding, micro_batch_size, adapter=True, temperature=1.0):
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
        with worker.ulysses_sharding_manager, context:
            with torch.no_grad():
                values, _ = worker.actor.compute_log_prob(data.to(torch.cuda.current_device()), calculate_entropy=False)
        return values[:count].detach().float().cpu()
    finally:
        worker.actor.use_remove_padding = previous
        if worker._is_offload_param:
            offload_fsdp_model_to_cpu(worker.actor_module_fsdp)
        torch.cuda.empty_cache()


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


def _make_batch(worker, collector, rows, observation):
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
    # Exactly the normal rollout worker method, including sharding-manager sync.
    output = worker.generate_sequences(generation)
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
        if "actor_training" in values:
            left, right = values["rollout"][index], values["actor_training"][index]
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
    config = OmegaConf.merge(
        OmegaConf.load(Path(verl.__file__).resolve().parent / "trainer/config/ppo_trainer.yaml"),
        OmegaConf.load(args.config), OmegaConf.from_dotlist(args.override),
    )
    OmegaConf.resolve(config)
    report["checkpoint_lora_config_source"] = "config YAML plus CLI overrides; .pt does not encode LoRA alpha"
    rollout = config.actor_rollout_ref.rollout
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
    rows = pq.read_table(args.parquet).to_pylist()[args.offset:args.offset + args.samples]
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
    if not args.skip_reference_second_turn:
        observation, _, done, _ = environment.step(_reference_actions(rows))
        if np.asarray(done).any():
            raise RuntimeError("Controlled tool requests did not produce answer-after-tool observations")
        observations.append(("reference_second_turn", observation))

    mixed_sources = []
    for name, observation in observations:
        print(f"Comparing {name}: {len(rows)} samples...", flush=True)
        case = {"name": name, "errors": {}, "comparisons": {}}
        report["cases"].append(case)
        batch, raw_prompts, images = _make_batch(worker, collector, rows, observation)
        generation_outputs = list(captured)
        case["generation_lora_ids"] = last_request["lora_ids"]
        case["generation_sampling"] = last_request["sampling"]
        case["generation_sync_event"] = len(report["sync_events"]) - 1
        values = {"rollout": batch.batch["rollout_log_probs"].float().cpu()}
        temperature = float(rollout.temperature)
        micro = int(rollout.log_prob_micro_batch_size_per_gpu)
        original_rmpad = bool(config.actor_rollout_ref.model.use_remove_padding)
        phases = [
            ("actor_training", dict(remove_padding=original_rmpad, micro_batch_size=micro, temperature=temperature)),
            ("actor_packed_single", dict(remove_padding=True, micro_batch_size=1, temperature=temperature)),
            ("actor_padded_single", dict(remove_padding=False, micro_batch_size=1, temperature=temperature)),
            ("actor_base", dict(remove_padding=False, micro_batch_size=1, adapter=False, temperature=1.0)),
        ]
        if temperature != 1.0:
            phases.append(("actor_raw", dict(remove_padding=original_rmpad, micro_batch_size=micro, temperature=1.0)))
        for phase, options in phases:
            print(f"  {phase}", flush=True)
            try:
                values[phase] = _actor_log_probs(worker, batch, **options)
            except Exception:
                case["errors"][phase] = traceback.format_exc()
                torch.cuda.empty_cache()
            persist()
        for phase, enabled in [("vllm_prefill", True), ("vllm_base_prefill", False)]:
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
        for left, right in comparisons:
            if left in values and right in values:
                case["comparisons"][f"{left}_vs_{right}"] = probability_difference(values[left], values[right], mask)
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
        try:
            values["actor_training"] = _actor_log_probs(worker, batch, remove_padding=original_rmpad,
                                                       micro_batch_size=micro, temperature=temperature)
        except Exception:
            case["errors"]["actor_training"] = traceback.format_exc()
            torch.cuda.empty_cache()
        mask = batch.batch["attention_mask"][:, -batch.batch["responses"].shape[1]:].bool().cpu()
        for left, right in [("rollout", "actor_training"), ("actor_training", "actor_separate_turns"),
                            ("actor_training", "actor_packed_single"), ("actor_training", "actor_padded_single")]:
            if left in values and right in values:
                case["comparisons"][f"{left}_vs_{right}"] = probability_difference(values[left], values[right], mask)
        case.update(rows=len(batch), response_tokens=int(mask.sum()), sampling_temperature=temperature,
                    actor_micro_batch_size=micro, actor_remove_padding=original_rmpad,
                    note="Reuses the exact responses above; only actor row grouping changes, interleaving one-image/two-image rows.")
        records = _sample_records(batch, images, generation_outputs, worker.tokenizer, values, args.top_tokens)
        case["generation_prompt_alignment_all_match"] = all(item["generation_prompt_alignment"]["matches"] for item in records)
        _write_json(args.output / "mixed_turns_samples.json", records)
        persist()

    final_lora = tensor_fingerprint(collect_lora_params(worker.actor_module_fsdp))
    report["final_actor_lora"] = final_lora
    report["actor_weights_unchanged"] = final_lora == report["loaded_actor_lora"]
    incomplete = (any(case["errors"] or any(value["nonfinite_tokens"] for value in case["comparisons"].values())
                      for case in report["cases"])
                  or any("inspection_error" in event for event in report["sync_events"]))
    report["status"] = "partial" if incomplete else "completed"


def _markdown_report(report):
    lines = ["# 训推一致性验证报告", "", f"状态：`{report['status']}`", "",
             f"Checkpoint：`{report['checkpoint']}`", "",
             "该脚本不更新参数，不保存或删除 checkpoint；读取完整 .pt 中的模型权重，不依赖独立 adapter 文件。", "",
             f"Checkpoint LoRA 与 actor 完全相同：`{report.get('checkpoint_lora_matches_actor', '未完成')}`。",
             f"检查前后 actor LoRA 未改变：`{report.get('actor_weights_unchanged', '未完成')}`。",
             f"源 checkpoint 文件大小与修改时间未改变：`{report.get('checkpoint_file_unchanged', '未完成')}`。", "",
             "`mean/std/max` 为同一批 response token 的概率绝对差；std 使用样本标准差，与训练指标一致。", "",
             "| 场景 | 对照 | mean | std | max | 缺失/非有限 token |", "|---|---|---:|---:|---:|---:|"]
    for case in report["cases"]:
        for name, item in case["comparisons"].items():
            numbers = [f"{item[key]:.6f}" if key in item else "n/a" for key in ("mean", "std", "max")]
            lines.append(f"| {case['name']} | {name} | {' | '.join(numbers)} | {item['nonfinite_tokens']} |")
    lines.extend(["", "## LoRA 同步检查", "",
                  "| 同步次数（0 为首次采样前） | 已核对/应核对矩阵 | GPU 槽位与 actor 一致 |", "|---|---:|---|"])
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
                  "- effective_config.yaml / run.log：复现配置与运行日志。", "",
                  "reference_second_turn 由参考框（缺失时用中心框）构造固定双图观察，专门检查第二轮；不是模型自然选择工具的比例。",
                  "各对照共享同一段已经采样的 response。没有把不同生成结果的概率直接相减。", "",
                  "LoRA rank、alpha、基座路径来自配置；.pt 无法自动提供 alpha，必须使用原训练值。",
                  "completed 表示所有诊断阶段执行完毕，并不表示两端概率一致。", "",
                  "缺失阶段和非有限 token 不代表通过。报告保留诊断数据，由人工结合各对照判断原因，不使用任意阈值宣布修复。", ""])
    if report.get("fatal_error"):
        lines.extend(["## 运行失败", "", "```text", report["fatal_error"], "```", ""])
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
