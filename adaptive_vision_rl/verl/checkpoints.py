"""Keep only the requested number of complete local DTPO checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


_STEP_DIR = re.compile(r"global_step_(\d+)")


def validate_lora_adapter(root: Path, step: int) -> int:
    """Check that the PEFT export contains tensor entries, not an empty header."""

    adapter = root / f"global_step_{step}" / "actor" / "lora_adapter"
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not config.is_file() or not weights.is_file():
        raise RuntimeError(
            f"LoRA adapter export is missing under {adapter}; the .pt checkpoint "
            "may still be resumable"
        )
    with weights.open("rb") as stream:
        size = weights.stat().st_size
        header_size = int.from_bytes(stream.read(8), "little")
        if header_size < 2 or header_size > min(size - 8, 16_000_000):
            raise RuntimeError(f"LoRA adapter has an invalid safetensors header: {weights}")
        header = json.loads(stream.read(header_size))
    tensor_count = sum(name != "__metadata__" for name in header)
    if not tensor_count:
        raise RuntimeError(
            f"LoRA adapter contains zero tensors: {weights}; the .pt checkpoint "
            "may still be resumable"
        )
    return tensor_count


def prune_local_checkpoints(root: Path, *, keep: int) -> list[Path]:
    """Clean old step directories after a successful save or resume.

    The pinned verl-agent tracks paths only in the current process and removes
    actor subdirectories, leaving old ``data.pt`` files after rotation. This
    handles both remnants and checkpoints from before a process restart.
    """

    if keep < 1:
        raise ValueError("keep must be positive")
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return []
    latest = int(marker.read_text().strip())
    candidates: list[tuple[int, Path]] = []
    complete: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _STEP_DIR.fullmatch(path.name)
        if not match or not path.is_dir() or path.is_symlink():
            continue
        step = int(match.group(1))
        candidates.append((step, path))
        actor = path / "actor"
        required = (
            actor / "model_world_size_1_rank_0.pt",
            actor / "optim_world_size_1_rank_0.pt",
            actor / "extra_state_world_size_1_rank_0.pt",
        )
        if step <= latest and all(item.is_file() for item in required):
            complete.append((step, path))

    if not any(step == latest for step, _ in complete):
        raise RuntimeError(f"latest checkpoint global_step_{latest} is incomplete")
    retained = {path for _, path in sorted(complete)[-keep:]}
    removed = []
    for _, path in candidates:
        if path not in retained:
            shutil.rmtree(path)
            removed.append(path)
    return removed


def install_checkpoint_retention(
    trainer, root: Path, *, keep: int, expect_lora: bool = False
) -> None:
    """Apply cross-restart retention while limiting peak disk use for keep=1."""

    original_load = trainer._load_checkpoint
    original_save = trainer._save_checkpoint
    saved_since_load = False

    def prune_or_warn():
        try:
            removed = prune_local_checkpoints(root, keep=keep)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Warning: DTPO checkpoint retention failed: {exc}", flush=True)
            return
        for path in removed:
            print(f"Removed old DTPO checkpoint: {path}", flush=True)

    def load_and_prune():
        nonlocal saved_since_load
        result = original_load()
        saved_since_load = False
        prune_or_warn()
        return result

    def save_and_prune():
        nonlocal saved_since_load
        # After a restart, verl-agent has no in-memory record of the loaded
        # checkpoint to rotate before writing the next one. Free it here so
        # the first resumed save fits in a one-checkpoint disk budget.
        if keep == 1 and not saved_since_load:
            marker = root / "latest_checkpointed_iteration.txt"
            if marker.is_file():
                previous = root / f"global_step_{int(marker.read_text().strip())}"
                if previous.is_dir() and not previous.is_symlink():
                    shutil.rmtree(previous)
                    print(
                        f"Removed old DTPO checkpoint before saving new one: {previous}",
                        flush=True,
                    )
        result = original_save()
        saved_since_load = True
        if expect_lora:
            tensor_count = validate_lora_adapter(root, int(trainer.global_steps))
            print(
                f"Validated DTPO LoRA adapter with {tensor_count} tensors "
                f"at global_step_{trainer.global_steps}",
                flush=True,
            )
        prune_or_warn()
        return result

    trainer._load_checkpoint = load_and_prune
    trainer._save_checkpoint = save_and_prune
