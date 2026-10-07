"""Save verl-agent's resumable checkpoint together with a validated adapter."""

from __future__ import annotations

from pathlib import Path

import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .lora_checkpoint import save_adapter
from .lora_sync import collect_lora_params


class DTPOActorRolloutRefWorker(ActorRolloutRefWorker):
    def _save_lora_adapter(self, local_path):
        adapter_path = Path(local_path) / "lora_adapter"
        tensors = None
        failure = None
        try:
            # All ranks must enter FSDP's parameter-gathering context.
            tensors = collect_lora_params(self.actor_module_fsdp)
        except Exception as exc:
            failure = f"rank {self.rank}: {type(exc).__name__}: {exc}"

        # A collection failure on any rank must not strand other ranks at a
        # later collective or publish an incomplete adapter on rank 0.
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        status = [None]
        if any(failures):
            status[0] = "; ".join(item for item in failures if item is not None)
        elif self.rank == 0:
            try:
                count = save_adapter(
                    adapter_path, tensors, self.actor_module.peft_config["default"]
                )
            except Exception as exc:
                status[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(status, src=0)
        if self.rank == 0:
            if status[0] is not None:
                print(
                    f"Warning: LoRA adapter export failed at {adapter_path}: {status[0]}. "
                    "Training will continue; the full .pt checkpoint is saved. "
                    "Export the adapter later with python -m scripts.export_dtpo_lora.",
                    flush=True,
                )
            else:
                print(f"[rank-0]: Saved {count} LoRA tensors to {adapter_path}", flush=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Save training state and the current LoRA adapter at the same step."""

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        try:
            self.checkpoint_manager.save_checkpoint(
                local_path=local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
            )
            dist.barrier()
            if self._is_lora:
                self._save_lora_adapter(local_path)

            if self.rank == 0:
                print(f"[rank-0]: Saved DTPO checkpoint at step {global_step}", flush=True)
        finally:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
