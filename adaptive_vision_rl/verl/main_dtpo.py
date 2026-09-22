"""Single-node DTPO entry point built on the pinned verl-agent trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from omegaconf import OmegaConf


def load_config(path: str, overrides: list[str]):
    import verl

    base = Path(verl.__file__).resolve().parent / "trainer/config/ppo_trainer.yaml"
    config = OmegaConf.merge(
        OmegaConf.load(base),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(overrides),
    )
    OmegaConf.resolve(config)
    return config


def run_dtpo(config):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    if not ray.is_initialized():
        default_runtime = get_ppo_ray_runtime_env()
        requested = config.get("ray_init", {}).get("runtime_env", {})
        runtime = OmegaConf.merge(default_runtime, requested)
        ray_kwargs = OmegaConf.create(
            {**config.get("ray_init", {}), "runtime_env": runtime}
        )
        ray.init(**OmegaConf.to_container(ray_kwargs, resolve=True))
    runner = DTPOTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class DTPOTaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.utils.vllm_utils import is_version_ge
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        from adaptive_vision_rl.environment import make_adaptive_vision_envs
        from adaptive_vision_rl.verl.collector import AdaptiveVisionTrajectoryCollector
        from adaptive_vision_rl.verl.dtpo import install_driver_hooks
        from adaptive_vision_rl.verl.reward_manager import AdaptiveVisionRewardManager

        pprint(OmegaConf.to_container(config, resolve=True))
        if config.trainer.n_gpus_per_node != 1 or config.trainer.nnodes != 1:
            raise ValueError("this project configuration currently targets one A800 GPU")
        if config.actor_rollout_ref.rollout.n != 1:
            raise ValueError("use env.rollout.n for DTPO groups; rollout.n must remain 1")
        if config.actor_rollout_ref.actor.entropy_coeff != 0:
            raise ValueError("entropy_coeff must be zero because loss_mask stores DTPO weights")
        if config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward:
            raise ValueError("the reproduced AdaptVision setup disables both KL paths")
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            raise ValueError("built-in tool multi_turn must stay disabled; the environment owns both turns")

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.trust_remote_code)
        processor = hf_processor(
            local_path,
            trust_remote_code=config.data.trust_remote_code,
            use_fast=True,
        )
        if processor is None:
            raise ValueError("Qwen3-VL processor is required")
        if config.actor_rollout_ref.model.lora_rank > 0 and not is_version_ge(
            pkg="vllm", minver="0.7.3"
        ):
            raise NotImplementedError("PPO LoRA requires vLLM >= 0.7.3")

        envs, val_envs = make_adaptive_vision_envs(config, processor)
        install_driver_hooks(config)

        actor_rollout_cls = ActorRolloutRefWorker
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        pool_name = "global_pool"
        resource_pool_spec = {pool_name: [1]}
        mapping = {Role.ActorRollout: pool_name, Role.Critic: pool_name}
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        reward_fn = AdaptiveVisionRewardManager(
            tokenizer,
            config.algorithm.dtpo,
            num_examine=0,
        )
        val_reward_fn = AdaptiveVisionRewardManager(
            tokenizer,
            config.algorithm.dtpo,
            num_examine=1,
        )
        collector = AdaptiveVisionTrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
        )
        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()

        # verl-agent's actor only selects loss_mask when this metadata switch is on.
        # It is enabled after trainer validation so the built-in tool engine remains off.
        config.actor_rollout_ref.rollout.multi_turn.enable = True
        trainer.fit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/dtpo_qwen3vl_4b_lora.yaml",
        help="Project override YAML merged on top of verl-agent ppo_trainer.yaml",
    )
    args, overrides = parser.parse_known_args()
    run_dtpo(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
