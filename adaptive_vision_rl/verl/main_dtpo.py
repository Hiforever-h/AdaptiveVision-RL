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


def run_dtpo(config, *, probe_steps=None, probe_output=None):
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
    ray.get(runner.run.remote(config, probe_steps=probe_steps, probe_output=probe_output))


@ray.remote(num_cpus=1)
class DTPOTaskRunner:
    def run(self, config, *, probe_steps=None, probe_output=None):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.utils.vllm_utils import is_version_ge
        from verl.workers.fsdp_workers import CriticWorker

        from adaptive_vision_rl.environment import make_adaptive_vision_envs
        from adaptive_vision_rl.thinking_template import configure_thinking_tokenizer
        from adaptive_vision_rl.verl.collector import AdaptiveVisionTrajectoryCollector
        from adaptive_vision_rl.verl.checkpoints import install_checkpoint_retention
        from adaptive_vision_rl.verl.dtpo import install_driver_hooks
        from adaptive_vision_rl.verl.reward_manager import AdaptiveVisionRewardManager
        from adaptive_vision_rl.verl.worker import DTPOActorRolloutRefWorker

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
        if config.env.max_steps != 2:
            raise ValueError("DTPO expects exactly two environment steps")
        actor = config.actor_rollout_ref.actor
        if actor.use_dynamic_bsz:
            raise ValueError("DTPO loss weights require fixed actor micro-batches")
        if actor.policy_loss.get("loss_mode", "vanilla") != "vanilla":
            raise ValueError("DTPO weighted policy loss requires vanilla PPO loss mode")
        trajectory_count = int(config.data.train_batch_size) * int(config.env.rollout.n)
        if int(actor.ppo_mini_batch_size) != 2 * trajectory_count:
            raise ValueError(
                "ppo_mini_batch_size must equal 2 * data.train_batch_size * "
                "env.rollout.n so both DTPO token denominators span the full step"
            )
        micro_batch_size = int(actor.ppo_micro_batch_size_per_gpu)
        if micro_batch_size < 1 or actor.ppo_mini_batch_size % micro_batch_size:
            raise ValueError("actor micro-batch size must divide ppo_mini_batch_size")
        log_prob_micro_batch_size = int(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
        )
        if (
            log_prob_micro_batch_size < 1
            or actor.ppo_mini_batch_size % log_prob_micro_batch_size
        ):
            raise ValueError("rollout log-prob micro-batch size must divide ppo_mini_batch_size")

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
        configure_thinking_tokenizer(tokenizer)
        if processor.tokenizer is not tokenizer:
            configure_thinking_tokenizer(processor.tokenizer)
        if config.actor_rollout_ref.model.lora_rank > 0 and not is_version_ge(
            pkg="vllm", minver="0.7.3"
        ):
            raise NotImplementedError("PPO LoRA requires vLLM >= 0.7.3")

        envs, val_envs = make_adaptive_vision_envs(config, processor)
        install_driver_hooks(config)

        actor_rollout_cls = DTPOActorRolloutRefWorker
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

        install_checkpoint_retention(
            trainer,
            Path(config.trainer.default_local_dir),
            keep=int(config.trainer.max_actor_ckpt_to_keep),
        )

        # verl-agent's actor only selects loss_mask when this metadata switch is on.
        # It is enabled after trainer validation so the built-in tool engine remains off.
        config.actor_rollout_ref.rollout.multi_turn.enable = True
        if probe_steps is None:
            trainer.fit()
        else:
            import json
            from adaptive_vision_rl.verl.short_run import install_short_run
            recorder, restore = install_short_run(trainer, steps=probe_steps, output=probe_output)
            try:
                (Path(probe_output) / "effective_config.json").write_text(
                    json.dumps(OmegaConf.to_container(config, resolve=True), ensure_ascii=False, indent=2) + "\n")
                trainer.fit()
                recorder.finish()
            except BaseException as exc:
                recorder.fail(exc)
                raise
            finally:
                restore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/dtpo_qwen3vl_4b_lora.yaml",
        help="Project override YAML merged on top of verl-agent ppo_trainer.yaml",
    )
    parser.add_argument("--probe-steps", type=int, choices=[10], help="Fresh 10-step diagnostic; preserve the full scheduler horizon")
    parser.add_argument("--probe-output", type=Path, help="Local directory for per-step metrics and completion state")
    args, overrides = parser.parse_known_args()
    if (args.probe_steps is None) != (args.probe_output is None):
        parser.error("--probe-steps and --probe-output must be supplied together")
    run_dtpo(load_config(args.config, overrides), probe_steps=args.probe_steps,
             probe_output=str(args.probe_output.resolve()) if args.probe_output is not None else None)


if __name__ == "__main__":
    main()
