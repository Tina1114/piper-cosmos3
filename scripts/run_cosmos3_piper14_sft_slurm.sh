#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-piper14-sft
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-external/cosmos/packages/cosmos3/.venv/bin/python}"
TOML_FILE="${TOML_FILE:-configs/cosmos3/sft/action_policy_piper14_nano.toml}"
PIPER14_ROOT="${PIPER14_ROOT:-/project/peilab/wam/physical_WM/data/battery_assemble/perfect}"
PIPER14_DATA_CONFIG="${PIPER14_DATA_CONFIG:-${REPO_ROOT}/configs/dataset_configs/battery_assemble_hdf5.yaml}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${REPO_ROOT}/reports/cosmos3_piper14}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/reports/cosmos3_piper14/matplotlib}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
QWEN_SNAPSHOT="${QWEN_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
HOME_DIR="${HOME_DIR:-${HOME:-${REPO_ROOT}}}"

if [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: missing Python executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${PIPER14_ROOT}" ]]; then
  echo "ERROR: missing PIPER14_ROOT directory: ${PIPER14_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${PIPER14_DATA_CONFIG}" ]]; then
  echo "ERROR: missing PIPER14_DATA_CONFIG: ${PIPER14_DATA_CONFIG}" >&2
  exit 2
fi
if [[ -z "${BASE_CHECKPOINT_PATH:-}" || ! -d "${BASE_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: set BASE_CHECKPOINT_PATH to the converted Cosmos3-Nano DCP checkpoint directory." >&2
  exit 2
fi
if [[ -z "${WAN_VAE_PATH:-}" || ! -f "${WAN_VAE_PATH}" ]]; then
  echo "ERROR: set WAN_VAE_PATH to Wan2.2_VAE.pth." >&2
  exit 2
fi
if [[ ! -d "${QWEN_SNAPSHOT}" ]]; then
  echo "ERROR: missing local Qwen snapshot: ${QWEN_SNAPSHOT}" >&2
  exit 2
fi

mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/uv"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/uv/tools"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/xdg"
mkdir -p "${REPO_ROOT}/external/cosmos/cache/imaginaire"
mkdir -p "${HOME_DIR}"
if [[ -n "${WANDB_API_KEY_FILE:-}" ]]; then
  if [[ ! -f "${WANDB_API_KEY_FILE}" ]]; then
    echo "ERROR: missing WANDB_API_KEY_FILE: ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
    if [[ -z "${WANDB_API_KEY}" ]]; then
      echo "ERROR: WANDB_API_KEY_FILE is empty: ${WANDB_API_KEY_FILE}" >&2
      exit 2
    fi
    export WANDB_API_KEY
  fi
fi
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
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PIPER14_ROOT
export PIPER14_DATA_CONFIG
export IMAGINAIRE_OUTPUT_ROOT
export MPLCONFIGDIR

TAIL_OVERRIDES=(
  ${EXTRA_TAIL_OVERRIDES:-}
)

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  -m piper_cosmos.cosmos3.train_action_policy_piper14 \
  --sft-toml="${TOML_FILE}" \
  "${TAIL_OVERRIDES[@]}"
