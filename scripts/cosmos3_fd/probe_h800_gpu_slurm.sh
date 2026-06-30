#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-gpu-probe
#SBATCH --output=reports/cosmos3_fd/gpu_probe_%j.out
#SBATCH --error=reports/cosmos3_fd/gpu_probe_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this script must be submitted with sbatch and run inside a SLURM job." >&2
  exit 2
fi

REPO_ROOT="${COSMOS3_REPO_ROOT:-$(pwd)}"
REPORT_DIR="${REPO_ROOT}/reports/cosmos3_fd"
FRAMEWORK_ROOT="${REPO_ROOT}/external/cosmos/packages/cosmos3"
PYTHON_BIN="${FRAMEWORK_ROOT}/.venv/bin/python"
LOG_PATH="${REPORT_DIR}/gpu_probe_${SLURM_JOB_ID}.log"

mkdir -p "${REPORT_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "started_at: $(date -Is)"
echo "slurm_job_id: ${SLURM_JOB_ID}"
echo "repo_root: ${REPO_ROOT}"
echo "pwd: $(pwd)"
echo "slurm_job_gpus: ${SLURM_JOB_GPUS:-unset}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "nvidia_smi:"
nvidia-smi

if [[ -x "${PYTHON_BIN}" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch

print("torch_version:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"device_{idx}_name:", torch.cuda.get_device_name(idx))
    print(f"device_{idx}_capability:", torch.cuda.get_device_capability(idx))
PY
else
  echo "missing framework python: ${PYTHON_BIN}"
fi

echo "finished_at: $(date -Is)"
echo "gpu_probe_log: ${LOG_PATH}"
