#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_DIR="${REPO_ROOT}/reports/cosmos3_fd"
NUM_GPUS="${COSMOS3_NUM_GPUS:-1}"

if [[ -z "${COSMOS3_SLURM_ACCOUNT:-}" ]]; then
  echo "ERROR: set COSMOS3_SLURM_ACCOUNT before submitting on this cluster." >&2
  exit 2
fi

if [[ -z "${COSMOS3_CHECKPOINT_PATH:-}" ]]; then
  echo "ERROR: set COSMOS3_CHECKPOINT_PATH before submitting DROID FD inference." >&2
  exit 3
fi

mkdir -p "${REPORT_DIR}"
sbatch \
  --account="${COSMOS3_SLURM_ACCOUNT}" \
  --chdir="${REPO_ROOT}" \
  --gres="gpu:${NUM_GPUS}" \
  --output="${REPORT_DIR}/slurm_%j.out" \
  --error="${REPORT_DIR}/slurm_%j.err" \
  --export=ALL,COSMOS3_REPO_ROOT="${REPO_ROOT}",COSMOS3_CHECKPOINT_PATH="${COSMOS3_CHECKPOINT_PATH}",COSMOS3_NUM_GPUS="${NUM_GPUS}",COSMOS3_DROID_NUM_CHUNKS="${COSMOS3_DROID_NUM_CHUNKS:-1}" \
  "${SCRIPT_DIR}/run_official_droid_fd_slurm.sh"
