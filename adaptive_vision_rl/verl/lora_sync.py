"""LoRA collection and first-rollout fixes for the pinned verl-agent release."""

from __future__ import annotations

from contextlib import nullcontext
from functools import wraps

from adaptive_vision_rl.verl.lora_checkpoint import validate_lora_tensors


def collect_lora_params(module) -> dict:
    """Collect owned CPU tensors while all FSDP1 parameters are materialized.

    The upstream layered collector assumes ``model.model.layers``, which skips
    Qwen3-VL's ``model.model.language_model.layers`` entirely. Summoning from
    the root also handles nested LoRA FSDP wrappers without hardcoded paths.
    """

    from peft.utils.save_and_load import get_peft_model_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    is_fsdp = isinstance(module, FSDP)
    model = module._fsdp_wrapped_module if is_fsdp else module
    context = FSDP.summon_full_params(module, writeback=False) if is_fsdp else nullcontext()
    with context:
        state = get_peft_model_state_dict(model, adapter_name="default")
        # Clone before leaving the context: CPU tensors can otherwise remain
        # views into FSDP storage that is freed or reshaped when it reshards.
        tensors = {}
        for name, value in state.items():
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            tensors[name] = value.detach().cpu().clone().contiguous()
        validate_lora_tensors(tensors, rank=model.peft_config["default"].r)

        expected = set()
        for name, child in model.named_modules():
            name = name.replace("_fsdp_wrapped_module.", "")
            for matrix in ("lora_A", "lora_B"):
                adapters = getattr(child, matrix, None)
                if adapters is not None and "default" in adapters:
                    expected.add(f"{name}.{matrix}.weight")
        missing = expected.difference(tensors)
        if missing:
            raise ValueError(f"Incomplete default LoRA state: missing {sorted(missing)}")
    return tensors


def _patch_initial_lora_sync(manager_cls: type) -> None:
    """Load both base weights and the current adapter before the first rollout."""

    original = manager_cls.update_params
    if getattr(original, "_dtpo_initial_lora_sync", False):
        return

    @wraps(original)
    def update_params(self, updated_params, peft_config=None):
        initial_lora_sync = peft_config is not None and not self.base_sync_done
        result = original(self, updated_params, peft_config=peft_config)
        if initial_lora_sync:
            # The upstream first call copies only base weights and then marks
            # base_sync_done. Its next call can now register the live adapter,
            # before __enter__ offloads the actor or wakes the KV cache.
            tensors = collect_lora_params(self.module)
            return original(self, tensors, peft_config=peft_config)
        return result

    update_params._dtpo_initial_lora_sync = True
    manager_cls.update_params = update_params


def install_lora_sync_fixes() -> None:
    """Install in the actor process via ``model.external_lib`` before rollout."""

    from verl.utils import fsdp_utils
    from verl.workers import fsdp_workers
    from verl.workers.sharding_manager import fsdp_vllm

    # These modules import the helper by value; patch every consumer as well
    # as its definition so both checkpoint export and layered rollout use it.
    for module in (fsdp_utils, fsdp_workers, fsdp_vllm):
        module.layered_summon_lora_params = collect_lora_params
    _patch_initial_lora_sync(fsdp_vllm.FSDPVLLMShardingManager)
