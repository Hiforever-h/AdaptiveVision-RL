"""Use verl-agent's resumable checkpoint without its optional LoRA export."""

from __future__ import annotations

import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class DTPOActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Save model, optimizer and scheduler; export an adapter offline when needed."""

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        dist.barrier()
        if self.rank == 0:
            print(
                f"[rank-0]: Saved DTPO checkpoint at step {global_step} "
                "without standalone LoRA adapter",
                flush=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
