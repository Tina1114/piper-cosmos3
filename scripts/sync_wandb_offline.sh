#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/external/cosmos/packages/cosmos3/.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_wandb_offline.sh <run_dir_or_offline_run_dir> [wandb sync args...]

Default W&B destination for the current login:
  Entity: nonoliu-harbin-institute-of-technology
  Project: cosmos3_action
  Project URL: https://wandb.ai/nonoliu-harbin-institute-of-technology/cosmos3_action

Examples:
  bash scripts/sync_wandb_offline.sh \
    reports/cosmos3_piper14/cosmos3_action/battery_piper14/my_run

  bash scripts/sync_wandb_offline.sh \
    reports/cosmos3_piper14/cosmos3_action/battery_piper14/my_run/wandb/offline-run-20260630_000000-abc12345

  bash scripts/sync_wandb_offline.sh <run_dir> --entity my-team --project cosmos3_action

After sync, view runs at:
  https://wandb.ai/<entity>/<project>
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: missing Python executable: ${PYTHON_BIN}" >&2
  exit 2
fi

INPUT_PATH="$1"
shift

if [[ "${INPUT_PATH}" != /* ]]; then
  INPUT_PATH="${REPO_ROOT}/${INPUT_PATH}"
fi

if [[ ! -e "${INPUT_PATH}" ]]; then
  echo "ERROR: path does not exist: ${INPUT_PATH}" >&2
  exit 2
fi

SYNC_TARGETS=()

if [[ -d "${INPUT_PATH}" && "$(basename "${INPUT_PATH}")" == offline-run-* ]]; then
  SYNC_TARGETS+=("${INPUT_PATH}")
elif [[ -d "${INPUT_PATH}/wandb" ]]; then
  while IFS= read -r path; do
    SYNC_TARGETS+=("${path}")
  done < <(find "${INPUT_PATH}/wandb" -maxdepth 1 -mindepth 1 -type d -name 'offline-run-*' | sort)
fi

if [[ ${#SYNC_TARGETS[@]} -eq 0 ]]; then
  echo "ERROR: no offline W&B runs found under: ${INPUT_PATH}" >&2
  echo "Expected either an offline-run-* directory or a run directory containing wandb/offline-run-*." >&2
  exit 2
fi

for target in "${SYNC_TARGETS[@]}"; do
  echo "Syncing ${target}"
  "${PYTHON_BIN}" -m wandb sync "${target}" "$@"
done
