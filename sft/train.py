#!/usr/bin/env python3
"""One A800 / one epoch Qwen3-VL-4B-Thinking LoRA SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_vision_rl.thinking_template import configure_thinking_tokenizer
from sft.data import Qwen3VLCollator, TurnDataset, encode_turn, load_turns, resolve_data_path


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if config["epochs"] != 1:
        raise ValueError("this SFT run must train exactly one epoch")
    if config["micro_batch_size"] < 1 or config["gradient_accumulation_steps"] < 1:
        raise ValueError("micro batch and gradient accumulation must be positive")
    if config["lora_rank"] < 1 or config["lora_alpha"] < 1:
        raise ValueError("invalid LoRA rank or alpha")
    return config


def resolve_data_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_preflight(rows: list[dict[str, Any]], processor: Any,
                    config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """Use the exact processor to reject overlength turns before loading the model."""
    max_prompt = (0, "")
    max_response = (0, "")
    max_total = (0, "")
    lengths_by_group: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        encoded = encode_turn(
            row, processor,
            max_prompt_tokens=config["max_prompt_tokens"],
            max_response_tokens=config["max_response_tokens"],
            data_dir=data_dir,
            include_pixels=False,
        )
        name = f"{row['sample_id']}:{row['turn_index']}"
        prompt_length = encoded["prompt_length"]
        response_length = encoded["response_length"]
        max_prompt = max(max_prompt, (prompt_length, name))
        max_response = max(max_response, (response_length, name))
        max_total = max(max_total, (prompt_length + response_length, name))
        lengths_by_group[f"{row['route']}/{row['stage']}"] += 1
        if index % 250 == 0:
            print(f"Token preflight {index}/{len(rows)}", flush=True)
    return {
        "turns": len(rows),
        "groups": dict(lengths_by_group),
        "max_prompt": {"tokens": max_prompt[0], "turn": max_prompt[1]},
        "max_response": {"tokens": max_response[0], "turn": max_response[1]},
        "max_total": {"tokens": max_total[0], "turn": max_total[1]},
    }


def make_trainer_class():
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class TurnMeanTrainer(Trainer):
        """Each turn contributes its mean assistant-token CE to the batch mean."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The custom loss is already averaged per micro batch. Trainer must
            # divide it by gradient_accumulation_steps during backward.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            prompt_lengths = inputs.pop("prompt_lengths")
            first_assistant = int(prompt_lengths.min().item())
            keep = inputs["input_ids"].shape[1] - first_assistant + 1
            outputs = model(**inputs, logits_to_keep=keep, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            targets = labels[:, first_assistant:]
            if logits.shape[:2] != targets.shape:
                raise RuntimeError("logits/label alignment failed")
            flat_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1), ignore_index=-100, reduction="none",
            ).reshape(targets.shape)
            valid = targets.ne(-100)
            token_counts = valid.sum(dim=1)
            if torch.any(token_counts == 0):
                raise RuntimeError("batch contains a turn with no supervised tokens")
            loss = (flat_loss.sum(dim=1) / token_counts).mean()
            return (loss, outputs) if return_outputs else loss

    return TurnMeanTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "sft/config.json")
    parser.add_argument("--model", help="Hub model ID or complete local model directory")
    parser.add_argument("--output-dir", type=Path, help="LoRA checkpoint output directory")
    parser.add_argument("--data-dir", type=Path, help="directory containing sft_adaptive_vision_v1/")
    parser.add_argument("--turns", type=Path, help="turns JSONL, absolute or relative to --data-dir")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    if args.preflight_only and args.skip_preflight:
        parser.error("--preflight-only and --skip-preflight are mutually exclusive")
    config = read_config(args.config)
    config["model"] = args.model or config["model"]
    config["output_dir"] = str(args.output_dir or config["output_dir"])
    config["data_dir"] = str(args.data_dir or config.get("data_dir", "data"))
    config["turns"] = str(args.turns or config["turns"])
    data_dir = resolve_data_dir(config["data_dir"])
    turns_path = resolve_data_path(config["turns"], data_dir)
    rows = load_turns(turns_path, data_dir=data_dir)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config["model"], use_fast=True)
    configure_thinking_tokenizer(processor.tokenizer)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        raise ValueError("Qwen3-VL tokenizer needs a pad token")
    if not args.skip_preflight:
        summary = token_preflight(rows, processor, config, data_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration, TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SFT requires exactly one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT requires native BF16 support")
    set_seed(config["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16,
        attn_implementation=config["attn_implementation"],
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    ))
    model.print_trainable_parameters()
    dataset = TurnDataset(
        rows, processor, config["max_prompt_tokens"], config["max_response_tokens"],
        data_dir=data_dir,
    )
    effective_batch = config["micro_batch_size"] * config["gradient_accumulation_steps"]
    total_steps = math.ceil(len(dataset) / effective_batch)
    half_step = round(len(dataset) / (2 * effective_batch))
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    if not args.resume_from_checkpoint and any(output_dir.glob("checkpoint-*")):
        raise FileExistsError(
            f"existing checkpoint in {output_dir}; use --resume-from-checkpoint or a new output_dir"
        )
    if not args.resume_from_checkpoint and (output_dir / "final").exists():
        raise FileExistsError(f"final adapter already exists in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "config": config,
        "turns_path": str(turns_path),
        "data_dir": str(data_dir),
        "turns_sha256": file_sha256(turns_path),
        "effective_batch_size": effective_batch,
        "expected_optimizer_steps": total_steps,
        "half_checkpoint_step": half_step,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    class HalfCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == half_step:
                control.should_save = True
            return control

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["micro_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=1.0,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        max_grad_norm=config["max_grad_norm"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        save_strategy="no",
        eval_strategy="no",
        logging_steps=5,
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        save_safetensors=True,
        seed=config["seed"],
        data_seed=config["seed"],
    )
    trainer = make_trainer_class()(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=Qwen3VLCollator(tokenizer.pad_token_id),
        callbacks=[HalfCheckpoint()],
        processing_class=tokenizer,
    )
    print(f"Training {len(dataset)} turns for one epoch; effective batch "
          f"{effective_batch}, half checkpoint step {half_step}, final step {total_steps}",
          flush=True)
    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint else None
    )
    if trainer.state.global_step != total_steps:
        raise RuntimeError(f"expected {total_steps} optimizer steps, got {trainer.state.global_step}")
    half_dir = output_dir / f"checkpoint-{half_step}"
    if not (half_dir / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"half-epoch adapter missing: {half_dir}")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    trainer.save_state()
    print(f"Half adapter: {half_dir}\nFinal adapter: {final_dir}", flush=True)


if __name__ == "__main__":
    main()
