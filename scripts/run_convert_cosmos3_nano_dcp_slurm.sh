#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-nano-dcp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-external/cosmos/packages/cosmos3/.venv/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Nano}"
QWEN_SNAPSHOT="${QWEN_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Nano-DCP}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/reports/cosmos3_piper14/convert_model_to_dcp.log}"
HOME_DIR="${HOME_DIR:-${HOME:-${REPO_ROOT}}}"

if [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
fi

mkdir -p "${REPO_ROOT}/reports/cosmos3_piper14"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/uv"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/uv/tools"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/xdg"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/imaginaire"
mkdir -p "${HOME_DIR}"

export HOME="${HOME_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/external/cosmos/cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-${REPO_ROOT}/external/cosmos/cache/uv/tools}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${REPO_ROOT}/external/cosmos/cache/xdg}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/external/cosmos/cache/xdg}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home}"
export IMAGINAIRE_CACHE_DIR="${IMAGINAIRE_CACHE_DIR:-${REPO_ROOT}/external/cosmos/cache/imaginaire}"
export COSMOS3_QWEN_SNAPSHOT="${COSMOS3_QWEN_SNAPSHOT:-${QWEN_SNAPSHOT}}"
export COSMOS3_WAN_VAE_PATH="${COSMOS3_WAN_VAE_PATH:-${WAN_VAE_PATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/cosmos/packages/cosmos3:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/convert_cosmos3_nano_to_dcp_offline.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --qwen-snapshot "${QWEN_SNAPSHOT}" \
  --wan-vae-path "${WAN_VAE_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  > "${LOG_PATH}" 2>&1

"${PYTHON_BIN}" scripts/verify_cosmos3_dcp.py \
  --path "${OUTPUT_PATH}" \
  --report "${REPO_ROOT}/reports/cosmos3_piper14/dcp_verify.json"
