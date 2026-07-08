#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-battery-export
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-external/cosmos/packages/cosmos3/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/cosmos_battery}"
STEPS="${STEPS:-12000 16000 18000 20000}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/reports/cosmos3_piper14/export_hf_logs}"
HOME_DIR="${HOME_DIR:-${HOME:-${REPO_ROOT}}}"
QWEN_SNAPSHOT="${QWEN_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"

if [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
fi

mkdir -p "${LOG_DIR}"
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/cosmos/packages/cosmos3:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/export_cosmos3_battery_batch.py \
  --output-root "${OUTPUT_ROOT}" \
  --steps ${STEPS}
