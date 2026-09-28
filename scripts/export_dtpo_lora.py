"""Export a DTPO LoRA adapter once, after training or from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from adaptive_vision_rl.verl.lora_checkpoint import export_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dtpo_qwen3vl_4b_lora.yaml"))
    parser.add_argument("--base-model", help="Override the merged SFT model path in the config")
    parser.add_argument("--rank", type=int, help="Override the training LoRA rank")
    parser.add_argument("--alpha", type=int, help="Override the training LoRA alpha")
    parser.add_argument("--expected-tensors", type=int)
    args = parser.parse_args()

    model = OmegaConf.load(args.config).actor_rollout_ref.model
    count = export_adapter(
        args.checkpoint,
        args.output,
        base_model=args.base_model or str(model.path),
        rank=args.rank if args.rank is not None else int(model.lora_rank),
        alpha=args.alpha if args.alpha is not None else int(model.lora_alpha),
        target_modules=list(model.target_modules),
        expected_tensors=args.expected_tensors,
    )
    print(f"Exported {count} LoRA tensors to {args.output.resolve()}")


if __name__ == "__main__":
    main()
