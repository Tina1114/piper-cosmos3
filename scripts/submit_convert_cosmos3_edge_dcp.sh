#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/reports/cosmos3_edge_piper14}"
PYTHON_BIN="${PYTHON_BIN:-/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3/.venv/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge-DCP}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
SLURM_ACCOUNT="${BATTERY_SLURM_ACCOUNT:-peilab}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
SLURM_QOS="${SLURM_QOS:-normal_qos}"

mkdir -p "${REPORT_DIR}"
for path in "${CHECKPOINT_PATH}" "${WAN_VAE_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required local asset: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_PATH}" ]] && find "${OUTPUT_PATH}" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: refusing to overwrite non-empty output: ${OUTPUT_PATH}" >&2
  exit 2
fi

sbatch \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --qos="${SLURM_QOS}" \
  --chdir="${REPO_ROOT}" \
  --output="${REPORT_DIR}/convert_dcp_%j.out" \
  --error="${REPORT_DIR}/convert_dcp_%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON_BIN="${PYTHON_BIN}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",OUTPUT_PATH="${OUTPUT_PATH}",WAN_VAE_PATH="${WAN_VAE_PATH}",REPORT_DIR="${REPORT_DIR}" \
  "${REPO_ROOT}/scripts/run_convert_cosmos3_edge_dcp_slurm.sh"
