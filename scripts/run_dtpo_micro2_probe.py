"""Replay real update-mode forwards, then train a fresh SFT LoRA for 10 steps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def short_config(source, output, *, base_model=None, data_root=None):
    config = copy.deepcopy(source)
    model = config["actor_rollout_ref"]["model"]
    if base_model:
        model["path"] = str(base_model)
    if data_root:
        config["env"]["adaptive_vision"]["data_root"] = str(data_root)
    actor = config["actor_rollout_ref"]["actor"]
    actor["ppo_micro_batch_size_per_gpu"] = 2
    rollout = config["actor_rollout_ref"]["rollout"]
    rollout["log_prob_micro_batch_size_per_gpu"] = 2
    # The diagnostic source used teacher-forcing headroom, not a training change.
    # All model precision, sampling, PPO and DTPO settings otherwise stay intact.
    rollout["multi_turn"]["enable"] = False
    trainer = config["trainer"]
    trainer.update(resume_mode="disable", resume_from_path=None, del_local_ckpt_after_load=False,
                   save_freq=-1, test_freq=-1, val_before_train=False, val_only=False,
                   logger=["console"], experiment_name=output.name + "_sft_micro2",
                   default_local_dir=str(output / "unused_checkpoints"), default_hdfs_dir=None,
                   rollout_data_dir=str(output / "train/rollouts"), validation_data_dir=None)
    return config


def run_logged(command, log_path, env):
    print("Running: " + " ".join(map(str, command)), flush=True)
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"Stage failed (exit {code}); see {log_path}")


def package(output):
    archive = output.with_name(output.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".yaml", ".md", ".log"):
                bundle.write(path, str(path.relative_to(output.parent)))
    return archive


def summary(output, state):
    lines = ["# micro=2 前向与 SFT 10 步短训", "", f"状态：`{state['status']}`", "",
             "第一阶段：使用 step 300 固定样本，比较旧概率 micro=4/2 与 train+grad+checkpointing 的 micro=2 前向。",
             "第二阶段：从 SFT 新建 LoRA，关闭自动恢复，旧概率和更新均使用 micro=2，训练 step 1–10。",
             "保留完整训练的 scheduler/warmup 初始化计划。关闭 checkpoint 保存及全量验证；console 与 metrics.jsonl 保存逐步指标。",
             "10 步用于检查运行和数值稳定性，不能证明最终精度或长期稳定性。", ""]
    metrics_path = output / "train/metrics.jsonl"
    if metrics_path.exists():
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        keys = ["training/rollout_probs_diff_mean", "training/rollout_probs_diff_std", "training/rollout_probs_diff_max",
                "actor/pg_clipfrac", "actor/ppo_kl", "actor/grad_norm", "actor/lr"]
        lines.extend(["| step | " + " | ".join(keys) + " |", "|---:|" + "---:|" * len(keys)])
        for record in records:
            if "training/global_step" not in record["metrics"]:
                continue
            data = record["metrics"]
            values = [f"{data[key]:.6g}" if isinstance(data.get(key), (int, float)) else "n/a" for key in keys]
            lines.append(f"| {record['step']} | " + " | ".join(values) + " |")
        lines.extend(["", "| step | accuracy | outcome reward | tool call rate | 非有限指标 |",
                      "|---:|---:|---:|---:|---|"])
        for record in records:
            data = record["metrics"]
            if "training/global_step" not in data:
                continue
            values = [f"{data[key]:.6g}" if isinstance(data.get(key), (int, float)) else "n/a"
                      for key in ("dtpo/accuracy", "dtpo/outcome_reward", "dtpo/tool_call_rate")]
            lines.append(f"| {record['step']} | " + " | ".join(values)
                         + " | " + ", ".join(record.get("nonfinite_metrics", [])) + " |")
    lines.extend(["", "前向结果：forward/REPORT.md、forward/report.json、forward/*_samples.json。",
                  "短训结果：train/run_state.json、train/metrics.jsonl、train/rollouts/、train.log。",
                  "启动配置：short_train_config.json；实际运行配置：train/effective_config.json；源报告校验信息：run.json。", ""])
    if state.get("error"):
        lines.extend(["```text", state["error"], "```", ""])
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_run2/global_step_300"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/outputs") / ("dtpo_micro2_step10_" + time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--parquet", type=Path, default=ROOT / "data/verl_agent/val.parquet")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    source_path = args.replay_report.expanduser().resolve()
    if source_path.is_dir():
        source_path = source_path / "report.json"
    source = json.loads(source_path.read_text())
    if source.get("status") != "completed" or not source.get("checkpoint_lora_matches_actor"):
        parser.error("Replay source must be a completed report with matching checkpoint weights")
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    base = (args.base_model or Path(source["effective_config"]["actor_rollout_ref"]["model"]["path"])).expanduser().resolve()
    # A run writes logs/configs. It must never share a tree with its inputs.
    for protected in (checkpoint, base, source_path.parent):
        if output == protected or output in protected.parents or protected in output.parents:
            parser.error(f"Output must be separate from checkpoint/model/source report: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", source_report=str(source_path),
                 source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                 checkpoint_for_forward_only=str(checkpoint), training_base_model=str(base),
                 training_start="fresh_sft", training_steps=10, project_commit=subprocess.run(
                     ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip())
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "third_party/verl-agent"), env.get("PYTHONPATH", "")]),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", VLLM_USE_V1="1",
               VLLM_ENABLE_V1_MULTIPROCESSING="0", RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="0")
    if not env.get("OMP_NUM_THREADS", "").isdigit() or int(env["OMP_NUM_THREADS"]) < 1:
        env["OMP_NUM_THREADS"] = "1"
    try:
        config = short_config(source["effective_config"], output, base_model=base, data_root=args.data_root)
        (output / "short_train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        command = [sys.executable, str(ROOT / "scripts/verify_rollout_consistency.py"),
                   "--checkpoint", str(checkpoint), "--replay-report", str(source_path),
                   "--output", str(output / "forward"), "--parquet", str(args.parquet),
                   "--base-model", str(base), "--update-forward-diagnostics"]
        if args.data_root:
            command.extend(["--data-root", str(args.data_root)])
        run_logged(command, output / "forward.log", env)
        forward = json.loads((output / "forward/report.json").read_text())
        if (forward["status"] != "completed" or not forward.get("actor_weights_unchanged")
                or not forward.get("checkpoint_file_unchanged")):
            raise RuntimeError("Forward probe incomplete or source weights changed; short training was not started")
        # Separate subprocesses ensure the forward probe's FSDP/vLLM CUDA memory
        # is released before Ray initializes fresh training workers.
        run_logged([sys.executable, "-m", "adaptive_vision_rl.verl.main_dtpo", "--config",
                    str(output / "short_train_config.json"), "--probe-steps", "10",
                    "--probe-output", str(output / "train")], output / "train.log", env)
        training = json.loads((output / "train/run_state.json").read_text())
        if training["status"] != "completed" or training["recorded_steps"] != list(range(1, 11)):
            raise RuntimeError("Training did not complete exactly steps 1–10")
        state["status"] = "completed"
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        print(state["error"], file=sys.stderr, flush=True)
    finally:
        (output / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        summary(output, state)
        archive = package(output)
        print(f"Report: {output / 'REPORT.md'}\nDownload this archive: {archive}", flush=True)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
