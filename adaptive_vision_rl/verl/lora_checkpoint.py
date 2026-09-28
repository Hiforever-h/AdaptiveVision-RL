"""Work around empty LoRA exports from the pinned verl-agent FSDP worker."""

from __future__ import annotations


def _collect_full_fsdp_lora_params(fsdp_module):
    from peft.utils.save_and_load import get_peft_model_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    peft_model = fsdp_module._fsdp_wrapped_module
    with FSDP.summon_full_params(fsdp_module, writeback=False):
        params = get_peft_model_state_dict(peft_model)
        return {
            name: (
                value.full_tensor() if hasattr(value, "full_tensor") else value
            ).detach().cpu().contiguous()
            for name, value in params.items()
        }


def collect_checkpoint_lora_params(
    fsdp_module, *, layered_collect, full_collect=None
):
    """Use the cheap layered path, then the full FSDP path if it found nothing."""

    params = layered_collect(fsdp_module)
    if not params:
        full_collect = full_collect or _collect_full_fsdp_lora_params
        params = full_collect(fsdp_module)
        print(
            f"Layered LoRA checkpoint extraction was empty; full FSDP extraction "
            f"found {len(params)} tensors",
            flush=True,
        )
    if not params:
        raise RuntimeError("LoRA checkpoint extraction found zero tensors")
    return {name: value.contiguous() for name, value in params.items()}


def install_lora_checkpoint_export_fix() -> None:
    """Replace only the helper used by the worker's adapter save path."""

    import verl.workers.fsdp_workers as fsdp_workers

    if getattr(fsdp_workers, "_adaptive_vision_lora_checkpoint_fix", False):
        return
    original = fsdp_workers.layered_summon_lora_params

    def collect(fsdp_module):
        return collect_checkpoint_lora_params(
            fsdp_module, layered_collect=original
        )

    fsdp_workers.layered_summon_lora_params = collect
    fsdp_workers._adaptive_vision_lora_checkpoint_fix = True
