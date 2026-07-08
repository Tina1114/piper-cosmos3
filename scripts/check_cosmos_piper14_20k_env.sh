#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_cosmos_piper14_20k.sh"

failures=0

check_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    printf '[ok] file %s\n' "${path}"
  else
    printf '[missing] file %s\n' "${path}"
    failures=$((failures + 1))
  fi
}

check_dir() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    printf '[ok] dir  %s\n' "${path}"
  else
    printf '[missing] dir  %s\n' "${path}"
    failures=$((failures + 1))
  fi
}

check_file "${CHECKPOINT_DIR}/config.json"
check_file "${CHECKPOINT_DIR}/checkpoint.json"
check_file "${CHECKPOINT_DIR}/model.safetensors.index.json"
check_file "${CONFIG_FILE}"
for shard in 1 2 3 4 5 6 7; do
  check_file "$(printf '%s/model-%05d-of-00007.safetensors' "${CHECKPOINT_DIR}" "${shard}")"
done

check_dir "${COSMOS3_QWEN_SNAPSHOT}"
check_file "${COSMOS3_QWEN_SNAPSHOT}/tokenizer.json"
check_file "${COSMOS3_QWEN_SNAPSHOT}/config.json"
check_file "${COSMOS3_WAN_VAE_PATH}"

printf '[info] HF_HOME=%s\n' "${HF_HOME}"
printf '[info] CONFIG_FILE=%s\n' "${CONFIG_FILE}"
printf '[info] HF_HUB_OFFLINE=%s\n' "${HF_HUB_OFFLINE}"
printf '[info] TRANSFORMERS_OFFLINE=%s\n' "${TRANSFORMERS_OFFLINE}"
printf '[info] PYTHONPATH=%s\n' "${PYTHONPATH}"
printf '[info] COSMOS_PIPER14_PYTHON=%s\n' "${COSMOS_PIPER14_PYTHON}"

"${COSMOS_PIPER14_PYTHON}" -c 'import importlib.util, os
mods = ["torch", "transformers", "safetensors", "numpy", "PIL", "piper_cosmos", "cosmos_framework"]
for mod in mods:
    spec = importlib.util.find_spec(mod)
    print("[python] {}: {}".format(mod, "ok" if spec else "missing"))
print("[python] HF_HOME={}".format(os.environ.get("HF_HOME")))'

"${COSMOS_PIPER14_PYTHON}" -c 'import json, os
with open(os.environ["CONFIG_FILE"], "r", encoding="utf-8") as f:
    data = json.load(f)
vae_path = data["model"]["config"]["tokenizer"]["vae_path"]
print("[python] config vae_path={}".format(vae_path))
print("[python] config vae_path_exists={}".format(os.path.isfile(vae_path)))'

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_query="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -n "${gpu_query}" ]]; then
    printf '[info] gpu %s\n' "${gpu_query}"
  else
    printf '[warn] nvidia-smi is installed but GPU query failed\n'
  fi
fi

if [[ ! -d "${COSMOS3_FRAMEWORK_ROOT}/cosmos_framework" ]]; then
  printf '[missing] Cosmos framework package at %s/cosmos_framework\n' "${COSMOS3_FRAMEWORK_ROOT}"
  failures=$((failures + 1))
fi

if [[ "${failures}" -gt 0 ]]; then
  printf '[result] incomplete: %d required filesystem checks failed\n' "${failures}"
  exit 1
fi

printf '[result] filesystem checks passed\n'
