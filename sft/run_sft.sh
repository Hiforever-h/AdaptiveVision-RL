#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "SFT training and vLLM evaluation require Linux with one CUDA GPU" >&2
  exit 1
fi

train_args=()
eval_args=()
while (($#)); do
  case "$1" in
    --config|--model|--data-dir)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      train_args+=("$1" "$2")
      eval_args+=("$1" "$2")
      shift 2
      ;;
    --output-dir)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      train_args+=("--output-dir" "$2")
      eval_args+=("--checkpoint-dir" "$2")
      shift 2
      ;;
    --turns)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      train_args+=("--turns" "$2")
      shift 2
      ;;
    --dataset-root)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      eval_args+=("--dataset-root" "$2")
      shift 2
      ;;
    --eval-output-dir)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      eval_args+=("--output-dir" "$2")
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

python -m sft.train "${train_args[@]}"
python -m sft.evaluate "${eval_args[@]}" --test
