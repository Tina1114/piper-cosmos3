#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${REPO_ROOT}/reports/battery_piper14"
SLURM_OUTPUT="${REPORT_DIR}/slurm_%j.out"
SLURM_ERROR="${REPORT_DIR}/slurm_%j.err"
PYTHON_BIN="${PYTHON_BIN:-external/cosmos/packages/cosmos3/.venv/bin/python}"
PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON:-python}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${REPORT_DIR}/preflight.json}"

mkdir -p "${REPORT_DIR}"

if [[ "${PYTHON_BIN}" = /* ]]; then
  PREFLIGHT_PYTHON_BIN="${PYTHON_BIN}"
else
  PREFLIGHT_PYTHON_BIN="${REPO_ROOT}/${PYTHON_BIN}"
fi

if ! "${PREFLIGHT_PYTHON}" "${REPO_ROOT}/scripts/preflight_battery_piper14_acceptance.py" \
  --python-bin "${PREFLIGHT_PYTHON_BIN}" \
  --config "${REPO_ROOT}/configs/train/battery_piper14.yaml" \
  --data-config "${REPO_ROOT}/configs/dataset_configs/battery_assemble_hdf5.yaml" \
  --safety-config "${REPO_ROOT}/configs/safety/battery_piper14_safety.yaml" \
  --require-slurm-account \
  --report "${PREFLIGHT_REPORT}"; then
  echo "ERROR: Battery Piper14 preflight failed. See ${PREFLIGHT_REPORT}" >&2
  echo "Example: BATTERY_SLURM_ACCOUNT=<your_PI_account> bash scripts/submit_battery_piper14_acceptance.sh" >&2
  exit 2
fi

sbatch \
  --account="${BATTERY_SLURM_ACCOUNT}" \
  --chdir="${REPO_ROOT}" \
  --output="${SLURM_OUTPUT}" \
  --error="${SLURM_ERROR}" \
  --export=ALL,PYTHON_BIN="${PYTHON_BIN}",MAX_STEPS="${MAX_STEPS:-500}",NUM_WORKERS="${NUM_WORKERS:-4}",EVAL_BATCHES="${EVAL_BATCHES:-64}",DEVICE="${DEVICE:-cuda}",SKIP_SAFETY_GATE="${SKIP_SAFETY_GATE:-1}" \
  "${REPO_ROOT}/scripts/run_battery_piper14_acceptance_slurm.sh"
