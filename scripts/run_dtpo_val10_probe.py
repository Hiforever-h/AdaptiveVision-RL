"""Validate SFT, train ten DTPO steps, validate again, and package the comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

from scripts.run_dtpo_micro2_probe import package


ROOT = Path(__file__).resolve().parents[1]


def build_command(config, output, overrides=(), *, base_model=None, data_root=None):
    # Passed last so a copied historical command cannot enable auto-resume,
    # checkpoints, intermediate validation, or different old/update batches.
    fixed = [
        "trainer.resume_mode=disable", "trainer.resume_from_path=null",
        "trainer.del_local_ckpt_after_load=false", "trainer.save_freq=-1",
        "trainer.test_freq=10", "trainer.val_before_train=true", "trainer.val_only=false",
        "trainer.critic_warmup=0", "trainer.logger=[console]", "trainer.log_val_generations=0",
        f"trainer.experiment_name={output.name}_sft_val10",
        f"trainer.default_local_dir={output / 'unused_checkpoints'}", "trainer.default_hdfs_dir=null",
        f"trainer.rollout_data_dir={output / 'train/rollouts'}", "trainer.validation_data_dir=null",
        "actor_rollout_ref.actor.ppo_micro_batch_size=null",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size=null",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.multi_turn.enable=false",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
    ]
    if base_model is not None:
        fixed.append(f"actor_rollout_ref.model.path={base_model}")
    if data_root is not None:
        fixed.append(f"env.adaptive_vision.data_root={data_root}")
    # Do not set trainer.total_training_steps here: workers must initialize
    # their optimizer scheduler using the full formal training horizon.
    return ["bash", str(ROOT / "scripts/run_dtpo_lora.sh"), "--config", str(config),
            "--probe-steps", "10", "--probe-output", str(output / "train"), "--probe-validation",
            *overrides, *fixed]


def exit_status(returncode, training):
    complete = (training.get("status") == "completed"
                and training.get("recorded_steps") == list(range(1, 11))
                and training.get("validation_steps") == [0, 10])
    signal_number = None
    if returncode is not None:
        # Bash reports a Python signal exit as 128 + signal number.
        candidate = -returncode if returncode < 0 else returncode-128
        if candidate in signal.valid_signals():
            signal_number = candidate
    return dict(status="completed" if returncode == 0 and complete else "failed",
                returncode=returncode, signal=signal.Signals(signal_number).name if signal_number else None,
                training_report_status=training.get("status", "missing"),
                recorded_steps=training.get("recorded_steps", []),
                validation_steps=training.get("validation_steps", []))


def write_report(output, state, training):
    lines = ["# SFT 验证 → 10 步更新 → 再验证", "", f"整体状态：`{state['status']}`；"
             f"训练进程退出码：`{state.get('returncode')}`；信号：`{state.get('signal')}`。", "",
             "从 SFT 基座新建 LoRA，第 0 步和第 10 步各跑一次完整验证集，均使用贪心生成。",
             "训练沿用正式配置，旧概率重算和更新 micro-batch 均为 2；不保存 checkpoint。",
             "保留正式训练的 scheduler 和 warmup 计划，只把训练循环的停止位置设为第 10 步。", "",
             f"已记录训练步骤：`{training.get('recorded_steps', [])}`；"
             f"已完成验证步骤：`{training.get('validation_steps', [])}`。", ""]
    if training:
        lines.append(f"完整学习率计划：{training.get('scheduler_horizon')} 步；"
                     f"warmup：{training.get('warmup_steps')} 步。")
    comparison_path = output / "train/validation_comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        lines.extend(["", "| 验证指标 | 更新前（step 0） | 更新后（step 10） | 后 − 前 |",
                      "|---|---:|---:|---:|"])
        for key, metric in comparison["metrics"].items():
            lines.append(f"| {key} | {metric['before']:.8g} | {metric['after']:.8g} | {metric['delta']:+.8g} |")
        lines.extend(["", f"对齐 {comparison['sample_count']} 道题："
                      f"输出 token 改变 {comparison['output_changed_count']} 题，"
                      f"工具使用改变 {comparison['tool_use_changed_count']} 题；"
                      f"错误→正确 {comparison['wrong_to_correct_count']} 题，"
                      f"正确→错误 {comparison['correct_to_wrong_count']} 题。", ""])
    else:
        lines.extend(["", "前后验证对比尚未完成；已完成的验证及训练记录仍保留。", ""])
    metrics_path = output / "train/metrics.jsonl"
    if metrics_path.exists():
        keys = ["actor/lr", "actor/grad_norm", "training/rollout_probs_diff_mean",
                "training/rollout_probs_diff_std", "training/rollout_probs_diff_max", "dtpo/accuracy"]
        lines.extend(["| step | " + " | ".join(keys) + " |", "|---:|" + "---:|" * len(keys)])
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            data = record["metrics"]
            if "training/global_step" not in data:
                continue
            values = [f"{data[key]:.6g}" if isinstance(data.get(key), (int, float)) else "n/a" for key in keys]
            lines.append(f"| {record['step']} | " + " | ".join(values) + " |")
    lines.extend(["", "10 步后指标可能不变，也可能变好或变差；不变本身不能证明权重没有更新。"
                  "结合逐步学习率、梯度范数和逐题输出变化判断，不能据此推断最终训练效果。", "",
                  "完整对比：train/validation_comparison.json；两次原始指标：train/validation_before.json、"
                  "train/validation_after.json；逐题输出：train/validation_*_samples.jsonl。",
                  "训练记录：train/metrics.jsonl、train/rollouts/；实际配置：train/effective_config.json；"
                  "日志：process.log；进程退出信息：run.json。", ""])
    if training.get("error"):
        lines.extend(["训练错误：", "```text", training["error"], "```", ""])
    if state.get("error"):
        lines.extend(["启动或进程错误：", "```text", state["error"], "```", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dtpo_qwen3vl_4b_lora.yaml")
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/outputs") /
                        ("dtpo_val10_" + time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--base-model", type=Path, help="Merged SFT base; starts a fresh LoRA without resuming run2")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("overrides", nargs="*", help="Formal training key=value overrides (retain the full scheduler horizon)")
    args = parser.parse_args()
    if args.base_model is not None:
        args.base_model = args.base_model.expanduser().resolve()
    if args.data_root is not None:
        args.data_root = args.data_root.expanduser().resolve()
    if any("=" not in value or value.startswith("-") for value in args.overrides):
        parser.error("Additional parameters must be key=value config overrides")
    output, config = args.output.expanduser().resolve(), args.config.expanduser().resolve()
    if not config.is_file():
        parser.error(f"Configuration does not exist: {config}")
    for protected in (ROOT, args.base_model, args.data_root):
        if protected is None:
            continue
        protected = Path(protected).expanduser().resolve()
        if output == protected or output in protected.parents or (protected != ROOT and protected in output.parents):
            parser.error(f"Output must be separate from input directory: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    command = build_command(config, output, args.overrides, base_model=args.base_model, data_root=args.data_root)
    shutil.copyfile(config, output / "source_config.yaml")
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
               PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""))
    versions = {}
    for name in ("torch", "transformers", "peft", "vllm", "ray"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    state = dict(status="running", command=command, python=sys.executable,
                 python_version=platform.python_version(), package_versions=versions,
                 project_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               capture_output=True, text=True).stdout.strip())
    (output / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    code, training = None, {}
    try:
        print("[val10] Fresh SFT LoRA: full val(step 0) → train(steps 1–10) → full val(step 10)", flush=True)
        with (output / "process.log").open("w", encoding="utf-8", buffering=1) as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
        if code != 0:
            state["error"] = f"Training process exited with code {code}; see process.log"
    except BaseException:
        state["error"] = traceback.format_exc()
        print(state["error"], file=sys.stderr, flush=True)
    finally:
        path = output / "train/run_state.json"
        if path.exists():
            try:
                training = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state["error"] = state.get("error", "") + "\nCannot read training state:\n" + traceback.format_exc()
        state.update(exit_status(code, training))
        (output / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_report(output, state, training)
        archive = package(output)
        print(f"Report: {output / 'REPORT.md'}\nDownload this archive: {archive}", flush=True)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
