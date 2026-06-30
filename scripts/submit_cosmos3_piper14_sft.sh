#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${REPO_ROOT}/reports/cosmos3_piper14"
SLURM_OUTPUT="${REPORT_DIR}/slurm_%j.out"
SLURM_ERROR="${REPORT_DIR}/slurm_%j.err"
PYTHON_BIN="${PYTHON_BIN:-external/cosmos/packages/cosmos3/.venv/bin/python}"
READINESS_PYTHON="${READINESS_PYTHON:-python}"
READINESS_REPORT="${READINESS_REPORT:-${REPORT_DIR}/readiness.json}"
TOML_FILE="${TOML_FILE:-configs/cosmos3/sft/action_policy_piper14_nano.toml}"
PIPER14_ROOT="${PIPER14_ROOT:-/project/peilab/wam/physical_WM/data/battery_assemble/perfect}"
PIPER14_DATA_CONFIG="${PIPER14_DATA_CONFIG:-${REPO_ROOT}/configs/dataset_configs/battery_assemble_hdf5.yaml}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${REPORT_DIR}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SLURM_GPUS="${SLURM_GPUS:-${NPROC_PER_NODE}}"
SLURM_CPUS="${SLURM_CPUS:-$(( SLURM_GPUS * 8 ))}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
DEFAULT_BASE_CHECKPOINT_PATH="${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Nano-DCP"
DEFAULT_WAN_VAE_PATH="${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"

mkdir -p "${REPORT_DIR}"

if [[ "${PYTHON_BIN}" = /* ]]; then
  READINESS_PYTHON_BIN="${PYTHON_BIN}"
else
  READINESS_PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
fi

if ! "${READINESS_PYTHON}" "${REPO_ROOT}/scripts/cosmos3_piper14_readiness.py" \
  --python-bin "${READINESS_PYTHON_BIN}" \
  --toml "${REPO_ROOT}/${TOML_FILE}" \
  --data-root "${PIPER14_ROOT}" \
  --data-config "${PIPER14_DATA_CONFIG}" \
  --base-checkpoint "${BASE_CHECKPOINT_PATH:-${DEFAULT_BASE_CHECKPOINT_PATH}}" \
  --wan-vae "${WAN_VAE_PATH:-${DEFAULT_WAN_VAE_PATH}}" \
  --output-root "${IMAGINAIRE_OUTPUT_ROOT}" \
  --require-slurm-account \
  --report "${READINESS_REPORT}"; then
  echo "ERROR: Cosmos3 Piper14 SFT readiness failed. See ${READINESS_REPORT}" >&2
  echo "Required: passed DCP verification, BATTERY_SLURM_ACCOUNT, BASE_CHECKPOINT_PATH, and WAN_VAE_PATH." >&2
  exit 2
fi

sbatch \
  --account="${BATTERY_SLURM_ACCOUNT}" \
  --chdir="${REPO_ROOT}" \
  --gres="gpu:${SLURM_GPUS}" \
  --cpus-per-task="${SLURM_CPUS}" \
  --time="${SLURM_TIME}" \
  --output="${SLURM_OUTPUT}" \
  --error="${SLURM_ERROR}" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON_BIN="${PYTHON_BIN}",TOML_FILE="${TOML_FILE}",PIPER14_ROOT="${PIPER14_ROOT}",PIPER14_DATA_CONFIG="${PIPER14_DATA_CONFIG}",BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}",WAN_VAE_PATH="${WAN_VAE_PATH}",IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT}",NPROC_PER_NODE="${NPROC_PER_NODE}" \
  "${REPO_ROOT}/scripts/run_cosmos3_piper14_sft_slurm.sh"
