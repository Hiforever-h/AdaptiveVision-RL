"""LoRA collection and rollout fixes for the pinned verl-agent release."""

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

    from peft.tuners.lora import LoraLayer
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    is_fsdp = isinstance(module, FSDP)
    model = module._fsdp_wrapped_module if is_fsdp else module
    context = FSDP.summon_full_params(module, writeback=False) if is_fsdp else nullcontext()
    with context:
        # An unwrapped PEFT model's state_dict() can expose flattened FSDP
        # parameters instead of named A/B weights. Read the actual LoRA layers
        # while their nested FSDP parameters are materialized instead.
        tensors = {}
        for name, child in model.named_modules():
            # FSDP delegates attribute access to its wrapped module. Restrict
            # this to actual LoRA layers to avoid collecting their aliases.
            if not isinstance(child, LoraLayer):
                continue
            if "default" not in child.r:
                continue
            has_a = "default" in child.lora_A
            has_b = "default" in child.lora_B
            if not has_a or not has_b:
                raise ValueError(f"Incomplete default LoRA state: missing A/B pair for {name}")
            name = ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")
            for matrix in ("lora_A", "lora_B"):
                key = f"{name}.{matrix}.weight"
                if key in tensors:
                    raise ValueError(f"Duplicate default LoRA tensor: {key}")
                value = getattr(child, matrix)["default"].weight
                if hasattr(value, "full_tensor"):
                    value = value.full_tensor()
                # Own storage before FSDP reshards, including when the source
                # is already on CPU and detach().cpu() would still be a view.
                tensors[key] = value.detach().cpu().clone().contiguous()
        validate_lora_tensors(tensors, rank=model.peft_config["default"].r)
    return tensors


def _patch_lora_collection(manager_cls: type) -> None:
    """Route every adapter collection through the same FSDP-safe collector."""

    original = manager_cls.__enter__
    if getattr(original, "_dtpo_lora_collection", False):
        return

    @wraps(original)
    def enter(self):
        previous = self.layered_summon
        if getattr(self.module, "peft_config", {}).get("default") is not None:
            # Upstream's layered branch calls our patched helper. The initial
            # dummy-weight sync must still use its base-model branch first.
            self.layered_summon = self.base_sync_done
        try:
            return original(self)
        finally:
            self.layered_summon = previous

    enter._dtpo_lora_collection = True
    manager_cls.__enter__ = enter


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
    _patch_lora_collection(fsdp_vllm.FSDPVLLMShardingManager)
    _patch_initial_lora_sync(fsdp_vllm.FSDPVLLMShardingManager)
