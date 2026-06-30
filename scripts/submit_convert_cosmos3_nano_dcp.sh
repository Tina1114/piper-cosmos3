#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${REPO_ROOT}/reports/cosmos3_piper14"
PYTHON_BIN="${PYTHON_BIN:-external/cosmos/packages/cosmos3/.venv/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Nano}"
QWEN_SNAPSHOT="${QWEN_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Nano-DCP}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_TIME="${SLURM_TIME:-08:00:00}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-192G}"

mkdir -p "${REPORT_DIR}"

if [[ -z "${BATTERY_SLURM_ACCOUNT:-}" ]]; then
  echo "ERROR: set BATTERY_SLURM_ACCOUNT before submitting the DCP conversion job." >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: missing local Cosmos3-Nano checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -d "${QWEN_SNAPSHOT}" ]]; then
  echo "ERROR: missing local Qwen snapshot: ${QWEN_SNAPSHOT}" >&2
  exit 2
fi
if [[ ! -f "${WAN_VAE_PATH}" ]]; then
  echo "ERROR: missing local Wan VAE file: ${WAN_VAE_PATH}" >&2
  exit 2
fi

sbatch \
  --account="${BATTERY_SLURM_ACCOUNT}" \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  --partition="${SLURM_PARTITION}" \
  --time="${SLURM_TIME}" \
  --cpus-per-task="${SLURM_CPUS}" \
  --mem="${SLURM_MEM}" \
  --chdir="${REPO_ROOT}" \
  --output="${REPORT_DIR}/convert_dcp_%j.out" \
  --error="${REPORT_DIR}/convert_dcp_%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON_BIN="${PYTHON_BIN}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",QWEN_SNAPSHOT="${QWEN_SNAPSHOT}",WAN_VAE_PATH="${WAN_VAE_PATH}",OUTPUT_PATH="${OUTPUT_PATH}" \
  "${REPO_ROOT}/scripts/run_convert_cosmos3_nano_dcp_slurm.sh"
