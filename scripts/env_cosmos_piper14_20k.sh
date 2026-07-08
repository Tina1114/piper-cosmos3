#!/usr/bin/env bash
# Source this file before starting the Cosmos Piper14 20k policy server.

export REPO_ROOT="/home/agilex/World_Action_Model/physical_WM/src/piper-cosmos3"
export CHECKPOINT_DIR="/home/agilex/World_Action_Model/physical_WM/checkpoints/cosmos_battery/20k"
export CONFIG_FILE="${REPO_ROOT}/configs/cosmos_piper14_20k_local_config.json"
export HF_HOME="/home/agilex/World_Action_Model/physical_WM/checkpoints/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export COSMOS3_QWEN_SNAPSHOT="${HF_HOME}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
export COSMOS3_WAN_VAE_PATH="/home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

export XDG_CACHE_HOME="/home/agilex/World_Action_Model/physical_WM/checkpoints/runtime_cache/xdg"
export XDG_DATA_HOME="/home/agilex/World_Action_Model/physical_WM/checkpoints/runtime_cache/xdg_data"
export IMAGINAIRE_CACHE_DIR="/home/agilex/World_Action_Model/physical_WM/checkpoints/runtime_cache/imaginaire"
export MPLCONFIGDIR="/home/agilex/World_Action_Model/physical_WM/checkpoints/runtime_cache/matplotlib"
export TMPDIR="/home/agilex/World_Action_Model/physical_WM/checkpoints/runtime_cache/tmp"

COSMOS3_FRAMEWORK_ROOT="${COSMOS3_FRAMEWORK_ROOT:-${REPO_ROOT}/external/cosmos}"
export COSMOS3_FRAMEWORK_ROOT
export PYTHONPATH="${REPO_ROOT}:${COSMOS3_FRAMEWORK_ROOT}"

export COSMOS_PIPER14_PYTHON="${COSMOS_PIPER14_PYTHON:-/home/agilex/miniconda3/envs/comos/bin/python}"
