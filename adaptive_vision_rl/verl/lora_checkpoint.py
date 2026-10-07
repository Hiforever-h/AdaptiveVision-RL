"""Save PEFT LoRA adapters from live actors or completed DTPO checkpoints."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping


def validate_lora_tensors(state: Mapping, *, rank: int) -> None:
    """Reject empty, unpaired, or sharded A/B tensors before using an adapter."""

    a_modules = set()
    b_modules = set()
    for name, value in state.items():
        if name.endswith(".lora_A.weight"):
            if len(value.shape) != 2 or value.shape[0] != rank or value.shape[1] <= 0:
                raise ValueError(f"Unexpected LoRA A shape for {name}: {value.shape}")
            a_modules.add(name.removesuffix(".lora_A.weight"))
        elif name.endswith(".lora_B.weight"):
            if len(value.shape) != 2 or value.shape[1] != rank or value.shape[0] <= 0:
                raise ValueError(f"Unexpected LoRA B shape for {name}: {value.shape}")
            b_modules.add(name.removesuffix(".lora_B.weight"))

    if rank <= 0 or not a_modules or a_modules != b_modules:
        raise ValueError(
            f"Incomplete default LoRA state: {len(a_modules)} A modules, "
            f"{len(b_modules)} B modules"
        )


def select_lora_tensors(state: Mapping, *, rank: int) -> dict:
    """Select default-adapter A/B weights and remove PEFT's runtime name."""

    result = {}
    for name, value in state.items():
        if name.endswith((".lora_A.default.weight", ".lora_B.default.weight")):
            exported = name.removesuffix(".default.weight") + ".weight"
        else:
            continue
        result[exported] = value

    validate_lora_tensors(result, rank=rank)
    return {name: value.detach().cpu().contiguous() for name, value in result.items()}


def save_adapter(output: Path, tensors: Mapping, config) -> int:
    """Publish an adapter directory only after weights and config are saved."""

    from safetensors.torch import save_file

    validate_lora_tensors(tensors, rank=config.r)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Adapter output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dtpo-adapter-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        save_file(dict(tensors), temporary_path / "adapter_model.safetensors")
        config.save_pretrained(temporary_path)
        temporary_path.rename(output)
    return len(tensors)


def export_adapter(
    checkpoint: Path,
    output: Path,
    *,
    base_model: str,
    rank: int,
    alpha: int,
    target_modules: list[str],
    expected_tensors: int | None = None,
) -> int:
    """Mmap the existing model .pt and write only its LoRA tensors."""

    import torch
    from peft import LoraConfig, TaskType

    checkpoint = Path(checkpoint)
    output = Path(output)
    model_pt = checkpoint / "actor" / "model_world_size_1_rank_0.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(model_pt)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Adapter output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    state = torch.load(model_pt, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a model state dictionary in {model_pt}")
    tensors = select_lora_tensors(state, rank=rank)
    if expected_tensors is not None and len(tensors) != expected_tensors:
        raise ValueError(
            f"Expected {expected_tensors} LoRA tensors, found {len(tensors)}"
        )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        bias="none",
    )
    config.base_model_name_or_path = base_model
    return save_adapter(output, tensors, config)
