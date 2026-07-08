#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${REPO_ROOT}/reports/cosmos3_piper14"
PYTHON_BIN="${PYTHON_BIN:-external/cosmos/packages/cosmos3/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/cosmos_battery}"
STEPS="${STEPS:-12000 16000 18000 20000}"
QWEN_SNAPSHOT="${QWEN_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_CPUS="${SLURM_CPUS:-16}"
SLURM_MEM="${SLURM_MEM:-256G}"

RUN_ROOT="${REPO_ROOT}/reports/cosmos3_piper14/cosmos3_action/battery_piper14/battery_piper14_cosmos3_nano_20000step_4gpu_b8_acc1_offline"
CONFIG_FILE="${RUN_ROOT}/config.yaml"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"

mkdir -p "${REPORT_DIR}"

if [[ -z "${BATTERY_SLURM_ACCOUNT:-}" ]]; then
  echo "ERROR: set BATTERY_SLURM_ACCOUNT before submitting the export job." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" && ! -x "${REPO_ROOT}/${PYTHON_BIN}" ]]; then
  echo "ERROR: missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: missing config file: ${CONFIG_FILE}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "ERROR: missing checkpoint root: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${QWEN_SNAPSHOT}" ]]; then
  echo "ERROR: missing local Qwen snapshot: ${QWEN_SNAPSHOT}" >&2
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
  --output="${REPORT_DIR}/export_battery_hf_%j.out" \
  --error="${REPORT_DIR}/export_battery_hf_%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON_BIN="${PYTHON_BIN}",OUTPUT_ROOT="${OUTPUT_ROOT}",STEPS="${STEPS}",QWEN_SNAPSHOT="${QWEN_SNAPSHOT}" \
  "${REPO_ROOT}/scripts/run_export_cosmos3_battery_hf_slurm.sh"
