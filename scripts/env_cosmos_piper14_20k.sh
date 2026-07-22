#!/usr/bin/env bash
# Source this file before starting the Cosmos Piper14 20k policy server.
# Resolve paths from this script so the same checkout works under both
# /home/agilex/... on the host and /workspace/... in the Docker container.

COSMOS_PIPER14_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${COSMOS_PIPER14_SCRIPT_DIR}/.." && pwd)"
export PHYSICAL_WM_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
export CHECKPOINT_DIR="${PHYSICAL_WM_ROOT}/checkpoints/cosmos_battery/20k"
export CONFIG_FILE="${REPO_ROOT}/configs/cosmos_piper14_20k_local_config.json"
export HF_HOME="${PHYSICAL_WM_ROOT}/checkpoints/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export COSMOS3_QWEN_SNAPSHOT="${HF_HOME}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
export COSMOS3_WAN_VAE_PATH="${PHYSICAL_WM_ROOT}/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

export XDG_CACHE_HOME="${PHYSICAL_WM_ROOT}/checkpoints/runtime_cache/xdg"
export XDG_DATA_HOME="${PHYSICAL_WM_ROOT}/checkpoints/runtime_cache/xdg_data"
export IMAGINAIRE_CACHE_DIR="${PHYSICAL_WM_ROOT}/checkpoints/runtime_cache/imaginaire"
export MPLCONFIGDIR="${PHYSICAL_WM_ROOT}/checkpoints/runtime_cache/matplotlib"
export TMPDIR="${PHYSICAL_WM_ROOT}/checkpoints/runtime_cache/tmp"

COSMOS3_FRAMEWORK_ROOT="${COSMOS3_FRAMEWORK_ROOT:-${REPO_ROOT}/external/cosmos}"
export COSMOS3_FRAMEWORK_ROOT
export PYTHONPATH="${REPO_ROOT}:${COSMOS3_FRAMEWORK_ROOT}"

COSMOS_PIPER14_ACTIVE_PYTHON="$(command -v python 2>/dev/null || true)"
if [[ -z "${COSMOS_PIPER14_ACTIVE_PYTHON}" && -x "/home/agilex/miniconda3/envs/cosmos/bin/python" ]]; then
  COSMOS_PIPER14_ACTIVE_PYTHON="/home/agilex/miniconda3/envs/cosmos/bin/python"
fi
if [[ -z "${COSMOS_PIPER14_ACTIVE_PYTHON}" && -x "${COSMOS3_FRAMEWORK_ROOT}/.venv-docker-cu130/bin/python" ]]; then
  COSMOS_PIPER14_ACTIVE_PYTHON="${COSMOS3_FRAMEWORK_ROOT}/.venv-docker-cu130/bin/python"
fi
export COSMOS_PIPER14_PYTHON="${COSMOS_PIPER14_PYTHON:-${COSMOS_PIPER14_ACTIVE_PYTHON}}"

if [[ ! -x "${COSMOS_PIPER14_PYTHON}" ]]; then
  printf '[error] COSMOS_PIPER14_PYTHON is not executable: %s\n' "${COSMOS_PIPER14_PYTHON}" >&2
  printf '[hint] Set COSMOS_PIPER14_PYTHON to the Python binary for the active Cosmos environment.\n' >&2
  return 1 2>/dev/null || exit 1
fi
