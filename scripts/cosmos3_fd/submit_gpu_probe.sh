#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_DIR="${REPO_ROOT}/reports/cosmos3_fd"

if [[ -z "${COSMOS3_SLURM_ACCOUNT:-}" ]]; then
  echo "ERROR: set COSMOS3_SLURM_ACCOUNT before submitting on this cluster." >&2
  echo "Example: COSMOS3_SLURM_ACCOUNT=<your_PI_account> bash scripts/cosmos3_fd/submit_gpu_probe.sh" >&2
  exit 2
fi

mkdir -p "${REPORT_DIR}"
sbatch \
  --account="${COSMOS3_SLURM_ACCOUNT}" \
  --chdir="${REPO_ROOT}" \
  --output="${REPORT_DIR}/gpu_probe_%j.out" \
  --error="${REPORT_DIR}/gpu_probe_%j.err" \
  --export=ALL,COSMOS3_REPO_ROOT="${REPO_ROOT}" \
  "${SCRIPT_DIR}/probe_h800_gpu_slurm.sh"
