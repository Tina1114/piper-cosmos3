#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_STAGE="${RUN_STAGE:-full}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/reports/cosmos3_edge_piper14}"
PYTHON_BIN="${PYTHON_BIN:-/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3/.venv/bin/python}"
TOML_FILE="${TOML_FILE:-configs/cosmos3/sft/action_policy_piper14_edge.toml}"
PIPER14_ROOT="${PIPER14_ROOT:-/project/peilab/wam/physical_WM/data/battery_assemble/perfect}"
PIPER14_DATA_CONFIG="${PIPER14_DATA_CONFIG:-${REPO_ROOT}/configs/dataset_configs/battery_assemble_hdf5.yaml}"
EDGE_SNAPSHOT="${EDGE_SNAPSHOT:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${REPO_ROOT}/external/cosmos/checkpoints/Cosmos3-Edge-DCP}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
NPROC_PER_NODE=4
SLURM_ACCOUNT="${BATTERY_SLURM_ACCOUNT:-peilab}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
SLURM_QOS="${SLURM_QOS:-normal_qos}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_MEM="${SLURM_MEM:-256G}"

case "${RUN_STAGE}" in
  smoke)
    RUN_NAME="battery_piper14_cosmos3_edge_base_fresh_head_smoke100"
    STAGE_OVERRIDES="job.name=${RUN_NAME} trainer.max_iter=100 scheduler.cycle_lengths=[100] checkpoint.save_iter=50"
    SLURM_TIME="${SLURM_TIME:-08:00:00}"
    REQUIRE_GATE=0
    EDGE_AUDIT_INFERENCE=0
    EDGE_AUDIT_EXPECT_RESUME_ITER=0
    ;;
  reload)
    # Reuse the smoke job name so DCP resolves and resumes its latest local
    # checkpoint. One extra optimizer step plus action sampling proves reload.
    RUN_NAME="battery_piper14_cosmos3_edge_base_fresh_head_smoke100"
    STAGE_OVERRIDES="job.name=${RUN_NAME} trainer.max_iter=102 scheduler.cycle_lengths=[102] checkpoint.save_iter=50"
    SLURM_TIME="${SLURM_TIME:-08:00:00}"
    REQUIRE_GATE=0
    EDGE_AUDIT_INFERENCE=1
    EDGE_AUDIT_EXPECT_RESUME_ITER=100
    ;;
  pilot2k)
    RUN_NAME="battery_piper14_cosmos3_edge_base_fresh_head_pilot2k"
    STAGE_OVERRIDES="job.name=${RUN_NAME} trainer.max_iter=2000 scheduler.cycle_lengths=[2000] checkpoint.save_iter=1000"
    SLURM_TIME="${SLURM_TIME:-24:00:00}"
    REQUIRE_GATE=1
    EDGE_AUDIT_INFERENCE=0
    EDGE_AUDIT_EXPECT_RESUME_ITER=0
    ;;
  pilot5k)
    RUN_NAME="battery_piper14_cosmos3_edge_base_fresh_head_pilot5k"
    STAGE_OVERRIDES="job.name=${RUN_NAME} trainer.max_iter=5000 scheduler.cycle_lengths=[5000] checkpoint.save_iter=1000"
    SLURM_TIME="${SLURM_TIME:-36:00:00}"
    REQUIRE_GATE=1
    EDGE_AUDIT_INFERENCE=0
    EDGE_AUDIT_EXPECT_RESUME_ITER=0
    ;;
  full)
    RUN_NAME="battery_piper14_cosmos3_edge_base_fresh_head_20k"
    STAGE_OVERRIDES="job.name=${RUN_NAME} trainer.max_iter=20000 scheduler.cycle_lengths=[20000] checkpoint.save_iter=500"
    SLURM_TIME="${SLURM_TIME:-72:00:00}"
    REQUIRE_GATE=1
    EDGE_AUDIT_INFERENCE=0
    EDGE_AUDIT_EXPECT_RESUME_ITER=0
    ;;
  *)
    echo "ERROR: RUN_STAGE must be smoke, reload, pilot2k, pilot5k, or full." >&2
    exit 2
    ;;
esac

mkdir -p "${REPORT_DIR}"
export BATTERY_SLURM_ACCOUNT="${SLURM_ACCOUNT}"
export BASE_CHECKPOINT_PATH WAN_VAE_PATH
export EXTRA_TAIL_OVERRIDES="${STAGE_OVERRIDES} ${EXTRA_TAIL_OVERRIDES:-}"
export EDGE_AUDIT_INFERENCE EDGE_AUDIT_EXPECT_RESUME_ITER
EDGE_TRAIN_AUDIT_REPORT="${REPORT_DIR}/runtime_audit_${RUN_STAGE}.json"
export EDGE_TRAIN_AUDIT_REPORT

READINESS_ARGS=(
  --python-bin "${PYTHON_BIN}"
  --toml "${REPO_ROOT}/${TOML_FILE}"
  --data-root "${PIPER14_ROOT}"
  --data-config "${PIPER14_DATA_CONFIG}"
  --edge-snapshot "${EDGE_SNAPSHOT}"
  --base-checkpoint "${BASE_CHECKPOINT_PATH}"
  --wan-vae "${WAN_VAE_PATH}"
  --output-root "${REPORT_DIR}"
  --require-slurm-account
  --report "${REPORT_DIR}/readiness_${RUN_STAGE}.json"
)
if [[ "${REQUIRE_GATE}" == "1" ]]; then
  READINESS_ARGS+=(--require-adversarial-gate)
fi

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/cosmos/packages/cosmos3" \
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/cosmos3_edge_piper14_readiness.py" "${READINESS_ARGS[@]}"

sbatch \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --qos="${SLURM_QOS}" \
  --gres="gpu:${NPROC_PER_NODE}" \
  --cpus-per-task="${SLURM_CPUS}" \
  --mem="${SLURM_MEM}" \
  --time="${SLURM_TIME}" \
  --chdir="${REPO_ROOT}" \
  --output="${REPORT_DIR}/${RUN_STAGE}_%j.out" \
  --error="${REPORT_DIR}/${RUN_STAGE}_%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON_BIN="${PYTHON_BIN}",TOML_FILE="${TOML_FILE}",PIPER14_ROOT="${PIPER14_ROOT}",PIPER14_DATA_CONFIG="${PIPER14_DATA_CONFIG}",EDGE_SNAPSHOT="${EDGE_SNAPSHOT}",BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}",WAN_VAE_PATH="${WAN_VAE_PATH}",IMAGINAIRE_OUTPUT_ROOT="${REPORT_DIR}",NPROC_PER_NODE="${NPROC_PER_NODE}",EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES}",EDGE_TRAIN_AUDIT_REPORT="${EDGE_TRAIN_AUDIT_REPORT}",EDGE_AUDIT_INFERENCE="${EDGE_AUDIT_INFERENCE}",EDGE_AUDIT_EXPECT_RESUME_ITER="${EDGE_AUDIT_EXPECT_RESUME_ITER}" \
  "${REPO_ROOT}/scripts/run_cosmos3_edge_piper14_sft_slurm.sh"
