#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_BIN="${HF_BIN:-/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3/.venv/bin/hf}"
EDGE_REPOSITORY="${EDGE_REPOSITORY:-nvidia/Cosmos3-Edge}"
EDGE_REVISION="${EDGE_REVISION:-6f58f6b4c91288838e60b6bcb2cc45d997e961de}"
EDGE_SNAPSHOT="${EDGE_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge}"

if [[ ! -x "${HF_BIN}" ]]; then
  echo "ERROR: missing Hugging Face CLI: ${HF_BIN}" >&2
  exit 2
fi
if [[ ! "${EDGE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: EDGE_REVISION must be an immutable 40-character commit SHA." >&2
  exit 2
fi

mkdir -p "$(dirname "${EDGE_SNAPSHOT}")"
"${HF_BIN}" download "${EDGE_REPOSITORY}" \
  --revision "${EDGE_REVISION}" \
  --local-dir "${EDGE_SNAPSHOT}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/cosmos/packages/cosmos3" \
  /project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3/.venv/bin/python - <<PY
from pathlib import Path
from piper_cosmos.cosmos3.local_edge_assets import REQUIRED_EDGE_FILES

root = Path("${EDGE_SNAPSHOT}")
missing = [str(root / relpath) for relpath in REQUIRED_EDGE_FILES if not (root / relpath).is_file()]
if missing:
    raise SystemExit("download incomplete; missing: " + ", ".join(missing))
print(f"verified fixed Cosmos3-Edge snapshot: {root}")
PY
