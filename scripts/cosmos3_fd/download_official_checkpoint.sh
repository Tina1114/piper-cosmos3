#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COSMOS_ROOT="${REPO_ROOT}/external/cosmos"
FRAMEWORK_ROOT="${COSMOS_ROOT}/packages/cosmos3"
CHECKPOINT_ROOT="${COSMOS3_CHECKPOINT_ROOT:-${REPO_ROOT}/external/cosmos/checkpoints}"
MODEL_ID="${COSMOS3_MODEL_ID:-nvidia/Cosmos3-Nano}"
PYTHON_BIN="${FRAMEWORK_ROOT}/.venv/bin/python"

unset_proxy_env() {
  unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset FTP_PROXY ftp_proxy WS_PROXY WSS_PROXY ws_proxy wss_proxy
  unset PIP_PROXY pip_proxy NPM_CONFIG_PROXY NPM_CONFIG_HTTP_PROXY NPM_CONFIG_HTTPS_PROXY
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: missing Cosmos Framework venv python: ${PYTHON_BIN}" >&2
  echo "Run scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh first." >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && ! -f "${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}" ]]; then
  echo "ERROR: set HF_TOKEN/HUGGING_FACE_HUB_TOKEN or login with the Hugging Face CLI before downloading gated official checkpoints." >&2
  exit 2
fi

mkdir -p "${CHECKPOINT_ROOT}"
source "${FRAMEWORK_ROOT}/.venv/bin/activate"
unset_proxy_env
export HF_HOME="${HF_HOME:-${CHECKPOINT_ROOT}/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${COSMOS_ROOT}/.uv_cache}"

checkpoint_path="${CHECKPOINT_ROOT}/${MODEL_ID##*/}"
"${FRAMEWORK_ROOT}/.venv/bin/hf" download "${MODEL_ID}" \
  --repo-type model \
  --revision main \
  --cache-dir "${HF_HOME}/hub" \
  --local-dir "${checkpoint_path}" \
  --max-workers "${COSMOS3_HF_DOWNLOAD_WORKERS:-8}"

echo "checkpoint_repo: ${MODEL_ID}"
echo "checkpoint_path: ${checkpoint_path}"
echo "Set COSMOS3_CHECKPOINT_PATH=${checkpoint_path}"
