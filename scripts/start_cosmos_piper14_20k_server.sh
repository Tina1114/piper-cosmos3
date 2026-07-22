#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_cosmos_piper14_20k.sh"

cd "${REPO_ROOT}"

exec "${COSMOS_PIPER14_PYTHON}" scripts/serve_cosmos_piper14_policy.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --config-file "${CONFIG_FILE}" \
  --host "${COSMOS_PIPER14_HOST:-0.0.0.0}" \
  --port "${COSMOS_PIPER14_PORT:-8766}" \
  "$@"
