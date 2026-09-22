#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_AGENT_ROOT="${VERL_AGENT_ROOT:-${PROJECT_ROOT}/third_party/verl-agent}"
EXPECTED_COMMIT="$(tr -d '[:space:]' < "${PROJECT_ROOT}/third_party/verl-agent.commit")"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "DTPO training requires a Linux CUDA host; macOS is supported for data and CPU checks only." >&2
  exit 1
fi

if [[ ! -d "${VERL_AGENT_ROOT}/.git" ]]; then
  echo "verl-agent checkout not found: ${VERL_AGENT_ROOT}" >&2
  echo "See docs/DTPO_VERL_AGENT.md for the pinned installation steps." >&2
  exit 1
fi

ACTUAL_COMMIT="$(git -C "${VERL_AGENT_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "verl-agent commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 1
fi

for split in train val; do
  if [[ ! -f "${PROJECT_ROOT}/data/verl_agent/${split}.parquet" ]]; then
    echo "missing data/verl_agent/${split}.parquet; run scripts/prepare_verl_data.py first" >&2
    exit 1
  fi
done

export PYTHONPATH="${PROJECT_ROOT}:${VERL_AGENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"

cd "${PROJECT_ROOT}"
python -m adaptive_vision_rl.verl.main_dtpo \
  --config configs/dtpo_qwen3vl_4b_lora.yaml \
  "$@"
