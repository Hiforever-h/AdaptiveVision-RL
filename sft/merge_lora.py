#!/usr/bin/env python3
"""Merge a Qwen3-VL LoRA adapter into its base model and export full weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_paths(adapter: Path, output_dir: Path, base_model: str) -> dict:
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"expected adapter_config.json and adapter_model.safetensors in {adapter}"
        )
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if output_dir == adapter or adapter in output_dir.parents:
        raise ValueError("output directory must be outside the adapter directory")
    local_base = Path(base_model).expanduser()
    if local_base.is_dir():
        base_dir = local_base.resolve()
        if output_dir == base_dir or base_dir in output_dir.parents:
            raise ValueError("output directory must be outside the local base model directory")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError(f"expected a LoRA adapter, got {adapter_config.get('peft_type')!r}")
    return adapter_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True,
                        help="Hub model ID or complete local base model directory")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="PEFT adapter directory, such as final/ or checkpoint-70/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for complete merged model weights")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    adapter = args.adapter.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    adapter_config = validate_paths(adapter, output_dir, args.base_model)

    import torch
    import peft
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only if enough RAM is available")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading base model: {args.base_model}", flush=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0" if args.device == "cuda" else "cpu"},
    )
    print(f"Loading LoRA adapter: {adapter}", flush=True)
    peft_model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    peft_model.eval()
    print("Merging adapter into base weights", flush=True)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True

    # Keep the base model's original processor. The project patches its
    # Thinking generation prefix at runtime when evaluating or rolling out.
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.merge-", dir=output_dir.parent))
    try:
        print(f"Saving full model to temporary directory: {temporary}", flush=True)
        merged.save_pretrained(
            str(temporary), safe_serialization=True, max_shard_size=args.max_shard_size
        )
        processor.save_pretrained(str(temporary))
        if not (temporary / "config.json").is_file():
            raise RuntimeError("merged model config was not saved")
        if not list(temporary.glob("*.safetensors")):
            raise RuntimeError("merged model weights were not saved")
        if (temporary / "adapter_config.json").exists():
            raise RuntimeError("export still contains a PEFT adapter config")
        manifest = {
            "base_model": args.base_model,
            "adapter": str(adapter),
            "adapter_base_model": adapter_config.get("base_model_name_or_path"),
            "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
            "dtype": args.dtype,
            "device_used_for_merge": args.device,
            "max_shard_size": args.max_shard_size,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "torch_version": torch.__version__,
        }
        (temporary / "merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"output appeared during export: {output_dir}")
        temporary.rename(output_dir)
        print(f"Merged model ready: {output_dir}", flush=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
