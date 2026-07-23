#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-edge-dcp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00

set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3/.venv/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge-DCP}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/reports/cosmos3_edge_piper14}"
HOME_DIR="${HOME_DIR:-${HOME:-${REPO_ROOT}}}"

for path in "${PYTHON_BIN}" "${WAN_VAE_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing required file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: missing local Cosmos3-Edge snapshot: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

mkdir -p "${REPORT_DIR}" "${REPO_ROOT}/external/cosmos/cache/uv/tools" \
  "${REPO_ROOT}/external/cosmos/cache/xdg" "${REPO_ROOT}/external/cosmos/cache/imaginaire"

export HOME="${HOME_DIR}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/external/cosmos/cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-${REPO_ROOT}/external/cosmos/cache/uv/tools}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${REPO_ROOT}/external/cosmos/cache/xdg}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/external/cosmos/cache/xdg}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home}"
export IMAGINAIRE_CACHE_DIR="${IMAGINAIRE_CACHE_DIR:-${REPO_ROOT}/external/cosmos/cache/imaginaire}"
export COSMOS3_EDGE_SNAPSHOT="${CHECKPOINT_PATH}"
export COSMOS3_WAN_VAE_PATH="${WAN_VAE_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/cosmos/packages/cosmos3:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/convert_cosmos3_edge_to_dcp_offline.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --wan-vae-path "${WAN_VAE_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  > "${REPORT_DIR}/convert_model_to_dcp.log" 2>&1

"${PYTHON_BIN}" scripts/verify_cosmos3_dcp.py \
  --path "${OUTPUT_PATH}" \
  --allow-missing-checkpoint-json \
  --report "${REPORT_DIR}/dcp_verify.json"
