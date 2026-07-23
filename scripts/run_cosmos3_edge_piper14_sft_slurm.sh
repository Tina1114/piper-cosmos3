#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-edge-piper14
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3/.venv/bin/python}"
TOML_FILE="${TOML_FILE:-configs/cosmos3/sft/action_policy_piper14_edge.toml}"
PIPER14_ROOT="${PIPER14_ROOT:-/project/peilab/wam/physical_WM/data/battery_assemble/perfect}"
PIPER14_DATA_CONFIG="${PIPER14_DATA_CONFIG:-${REPO_ROOT}/configs/dataset_configs/battery_assemble_hdf5.yaml}"
EDGE_SNAPSHOT="${EDGE_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge-DCP}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${REPO_ROOT}/reports/cosmos3_edge_piper14}"
EDGE_TRAIN_AUDIT_REPORT="${EDGE_TRAIN_AUDIT_REPORT:-${IMAGINAIRE_OUTPUT_ROOT}/runtime_audit_${SLURM_JOB_ID:-local}.json}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${IMAGINAIRE_OUTPUT_ROOT}/matplotlib}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
HOME_DIR="${HOME_DIR:-${HOME:-${REPO_ROOT}}}"

if [[ "${NPROC_PER_NODE}" != "4" ]]; then
  echo "ERROR: audited Edge baseline requires NPROC_PER_NODE=4, got ${NPROC_PER_NODE}." >&2
  exit 2
fi
for path in "${PYTHON_BIN}" "${PIPER14_DATA_CONFIG}" "${WAN_VAE_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing required file: ${path}" >&2
    exit 2
  fi
done
for path in "${PIPER14_ROOT}" "${EDGE_SNAPSHOT}" "${BASE_CHECKPOINT_PATH}"; do
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing required directory: ${path}" >&2
    exit 2
  fi
done

if [[ -n "${WANDB_API_KEY_FILE:-}" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ ! -s "${WANDB_API_KEY_FILE}" ]]; then
    echo "ERROR: WANDB_API_KEY_FILE is missing or empty: ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
  export WANDB_API_KEY
fi
if [[ -z "${WANDB_API_KEY:-}" && ! -f "${HOME_DIR}/.netrc" ]]; then
  echo "ERROR: W&B online mode needs WANDB_API_KEY, WANDB_API_KEY_FILE, or ${HOME_DIR}/.netrc." >&2
  exit 2
fi

mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}" "${MPLCONFIGDIR}" \
  "${REPO_ROOT}/external/cosmos/cache/uv/tools" "${REPO_ROOT}/external/cosmos/cache/xdg" \
  "${REPO_ROOT}/external/cosmos/cache/imaginaire"

export HOME="${HOME_DIR}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/external/cosmos/cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-${REPO_ROOT}/external/cosmos/cache/uv/tools}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${REPO_ROOT}/external/cosmos/cache/xdg}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/external/cosmos/cache/xdg}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home}"
export IMAGINAIRE_CACHE_DIR="${IMAGINAIRE_CACHE_DIR:-${REPO_ROOT}/external/cosmos/cache/imaginaire}"
export COSMOS3_EDGE_SNAPSHOT="${EDGE_SNAPSHOT}"
export COSMOS3_WAN_VAE_PATH="${WAN_VAE_PATH}"
export WAN_VAE_PATH
export BASE_CHECKPOINT_PATH
export PIPER14_ROOT
export PIPER14_DATA_CONFIG
export IMAGINAIRE_OUTPUT_ROOT
export EDGE_TRAIN_AUDIT_REPORT
export MPLCONFIGDIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/cosmos/packages/cosmos3:${PYTHONPATH:-}"

read -r -a TAIL_OVERRIDES <<< "${EXTRA_TAIL_OVERRIDES:-}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + ${SLURM_JOB_ID:-1} % 20000 ))}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m piper_cosmos.cosmos3.train_action_policy_piper14_edge \
  --sft-toml="${TOML_FILE}" \
  "${TAIL_OVERRIDES[@]}"
