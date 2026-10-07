"""Forward-only residual diagnostics, with a parent process preserving crash logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def exit_status(returncode, report):
    complete = (report.get("status") == "completed" and report.get("actor_weights_unchanged")
                and report.get("checkpoint_file_unchanged") and report.get("shutdown", {}).get("status") == "completed")
    return dict(status="completed" if returncode == 0 and complete else "failed",
                returncode=returncode, signal=signal.Signals(-returncode).name if returncode is not None and returncode < 0 else None,
                numerical_report_status=report.get("status", "missing"), training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", type=Path, required=True,
                        help="Completed update-forward report directory, e.g. dtpo_micro2_step10_.../forward")
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_run2/global_step_300"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/outputs") / ("rollout_residual_step300_" + time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--parquet", type=Path, default=ROOT / "data/verl_agent/val.parquet")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    source = args.replay_report.expanduser().resolve()
    source = source / "report.json" if source.is_dir() else source
    previous = json.loads(source.read_text())
    if (previous.get("status") != "completed" or not previous.get("update_forward_diagnostics")
            or not previous.get("checkpoint_lora_matches_actor")):
        parser.error("Use the completed forward/report.json from the micro=2 update-mode verification")
    output = args.output.expanduser().resolve()
    base = args.base_model or Path(previous["effective_config"]["actor_rollout_ref"]["model"]["path"])
    for path in (source.parent, args.checkpoint.expanduser().resolve(), base.expanduser().resolve()):
        if output == path or output in path.parents or path in output.parents:
            parser.error(f"Output must be separate from source/checkpoint/base model: {path}")
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-X", "faulthandler", str(ROOT / "scripts/verify_rollout_consistency.py"),
               "--replay-report", str(source), "--checkpoint", str(args.checkpoint),
               "--output", str(output / "forward"), "--parquet", str(args.parquet), "--residual-forward-diagnostics"]
    for flag, value in (("--base-model", args.base_model), ("--data-root", args.data_root)):
        if value is not None:
            command.extend([flag, str(value)])
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    code, forward, error = None, {}, None
    try:
        with (output / "process.log").open("w", buffering=1) as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
        report_path = output / "forward/report.json"
        if report_path.is_file():
            forward = json.loads(report_path.read_text())
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
    finally:
        state = exit_status(code, forward)
        state.update(command=command, source_report=str(source), error=error)
        (output / "process_exit.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        text = ["# 生成端残余概率差诊断", "", f"整体状态：`{state['status']}`；子进程退出码：`{code}`；"
                f"信号：`{state['signal']}`；数值报告状态：`{state['numerical_report_status']}`。", "",
                "本入口只执行固定样本前向，没有训练、反传或 checkpoint 保存。",
                "子进程异常退出也会保留已完成的结果和原生错误日志，不将非零退出码当作成功。", "",
                "前向对照：forward/REPORT.md、forward/report.json、forward/*_samples.json。",
                "进程日志（含 faulthandler）：process.log；退出信息：process_exit.json。", ""]
        if forward:
            text.append((output / "forward/REPORT.md").read_text() if (output / "forward/REPORT.md").exists() else "数值报告未完成。")
        if error:
            text.extend(["```text", error, "```"])
        (output / "REPORT.md").write_text("\n".join(text))
        archive = output.with_name(output.name + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".yaml", ".md", ".log"):
                    bundle.write(path, str(path.relative_to(output.parent)))
        print(f"Report: {output / 'REPORT.md'}\nDownload this archive: {archive}", flush=True)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
