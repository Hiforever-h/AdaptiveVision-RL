"""Compare a saved DTPO checkpoint through the original FSDP and vLLM paths."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / "third_party/verl-agent/verl").is_dir():
    sys.path.insert(1, str(ROOT / "third_party/verl-agent"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_run2/global_step_300"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dtpo_qwen3vl_4b_lora.yaml")
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/outputs") / ("rollout_consistency_step300_" + time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--parquet", type=Path, default=ROOT / "data/verl_agent/val.parquet")
    parser.add_argument("--data-root", type=Path, help="Override the image dataset root")
    parser.add_argument("--base-model", help="Original merged SFT model directory used by training")
    parser.add_argument("--samples", type=int, help="Rows to check in each one-image/two-image scenario (default: 4)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--top-tokens", type=int, default=20)
    parser.add_argument("--max-response-tokens", type=int, help="Optional shorter response limit; default preserves the training setting")
    parser.add_argument("--skip-reference-second-turn", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--replay-report", type=Path, help="Reuse the saved configuration, samples and responses from an earlier report directory")
    parser.add_argument("--forward-diagnostics", action="store_true", help="Replay focus tokens with repeated, base-only and precision-controlled actor forwards")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE", help="Training config override; repeat for multiple keys")
    args = parser.parse_args()
    if args.forward_diagnostics and args.replay_report is None:
        parser.error("--forward-diagnostics requires --replay-report to preserve the anomalous tokens")
    if args.replay_report and (args.samples is not None or args.offset or args.override or args.max_response_tokens is not None):
        parser.error("Replay preserves the original samples/configuration; omit samples/offset/override/max-response-tokens")
    if args.samples is None:
        args.samples = 4
    if args.samples < 1 or args.offset < 0 or args.top_tokens < 1:
        parser.error("samples/top-tokens must be positive and offset must be nonnegative")
    if args.max_response_tokens is not None and args.max_response_tokens < 1:
        parser.error("max-response-tokens must be positive")
    from adaptive_vision_rl.verl.consistency import run_verification
    raise SystemExit(run_verification(args))


if __name__ == "__main__":
    main()
