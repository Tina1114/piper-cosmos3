#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COSMOS_ROOT="${REPO_ROOT}/external/cosmos"
FRAMEWORK_ROOT="${COSMOS_ROOT}/packages/cosmos3"
COSMOS_TOOLS_DIR="${COSMOS_ROOT}/.tools"
REPORT_DIR="${REPO_ROOT}/reports/cosmos3_fd"
UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"
CONDA_ENV_NAME="${COSMOS3_CONDA_ENV_NAME:-cosmos3}"
CONDA_ENV_PREFIX="${COSMOS3_CONDA_ENV_PREFIX:-${COSMOS_ROOT}/conda_envs/${CONDA_ENV_NAME}}"

mkdir -p "${REPORT_DIR}"

unset_proxy_env() {
  unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset FTP_PROXY ftp_proxy WS_PROXY WSS_PROXY ws_proxy wss_proxy
  unset PIP_PROXY pip_proxy NPM_CONFIG_PROXY NPM_CONFIG_HTTP_PROXY NPM_CONFIG_HTTPS_PROXY
}

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required by the official Cosmos cookbook but was not found on PATH." >&2
  exit 1
fi

if [[ "${COSMOS3_USE_CONDA_ENV:-1}" == "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
    unset_proxy_env
    conda create -y -p "${CONDA_ENV_PREFIX}" python=3.13
  fi
  conda activate "${CONDA_ENV_PREFIX}"
fi

if ! command -v uv >/dev/null 2>&1 && [[ "${COSMOS3_BOOTSTRAP_UV:-0}" == "1" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: COSMOS3_BOOTSTRAP_UV=1 requires curl, but curl was not found on PATH." >&2
    exit 1
  fi
  mkdir -p "${COSMOS_TOOLS_DIR}"
  echo "uv not found on PATH; installing uv into isolated tools dir: ${COSMOS_TOOLS_DIR}"
  unset_proxy_env
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${COSMOS_TOOLS_DIR}" sh
  export PATH="${COSMOS_TOOLS_DIR}:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required by the official Cosmos cookbook but was not found on PATH." >&2
  echo "Install/load uv, or rerun with COSMOS3_BOOTSTRAP_UV=1 to install uv under external/cosmos/.tools." >&2
  echo "No base/fastwam packages were changed." >&2
  exit 1
fi

if [[ ! -d "${COSMOS_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${COSMOS_ROOT}")"
  unset_proxy_env
  git clone https://github.com/NVIDIA/cosmos.git "${COSMOS_ROOT}"
fi

if [[ ! -d "${FRAMEWORK_ROOT}/.git" ]]; then
  mkdir -p "${COSMOS_ROOT}/packages"
  unset_proxy_env
  git clone https://github.com/NVIDIA/cosmos-framework.git "${FRAMEWORK_ROOT}"
fi

{
  echo "prepared_at: $(date -Is)"
  echo "cosmos_root: ${COSMOS_ROOT}"
  echo "cosmos_commit: $(git -C "${COSMOS_ROOT}" rev-parse HEAD)"
  echo "framework_root: ${FRAMEWORK_ROOT}"
  echo "framework_commit: $(git -C "${FRAMEWORK_ROOT}" rev-parse HEAD)"
  echo "uv_group: ${UV_GROUP}"
  echo "conda_env_prefix: ${CONDA_ENV_PREFIX}"
} | tee "${REPORT_DIR}/cosmos3_fd_env_commits.txt"

cd "${FRAMEWORK_ROOT}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
unset_proxy_env
uv sync --all-extras --group="${UV_GROUP}"

echo "Cosmos Framework venv prepared at ${FRAMEWORK_ROOT}/.venv"
