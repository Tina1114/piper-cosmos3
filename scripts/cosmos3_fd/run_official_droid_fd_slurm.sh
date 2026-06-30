#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-droid-fd
#SBATCH --output=reports/cosmos3_fd/slurm_%j.out
#SBATCH --error=reports/cosmos3_fd/slurm_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this script must be submitted with sbatch and run inside a SLURM job." >&2
  exit 2
fi

REPO_ROOT="${COSMOS3_REPO_ROOT:-$(pwd)}"
COSMOS_ROOT="${REPO_ROOT}/external/cosmos"
FRAMEWORK_ROOT="${COSMOS_ROOT}/packages/cosmos3"
PYTHON_BIN="${FRAMEWORK_ROOT}/.venv/bin/python"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${REPO_ROOT}/reports/cosmos3_fd/outputs/${TIMESTAMP}"
INPUT_DIR="${RUN_ROOT}/inputs"
OUTPUT_DIR="${RUN_ROOT}/action_forward_dynamics_robotics_custom"
LOG_PATH="${RUN_ROOT}/run.log"
NUM_CHUNKS="${COSMOS3_DROID_NUM_CHUNKS:-1}"
NUM_GPUS="${COSMOS3_NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

unset_proxy_env() {
  unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset FTP_PROXY ftp_proxy WS_PROXY WSS_PROXY ws_proxy wss_proxy
  unset PIP_PROXY pip_proxy NPM_CONFIG_PROXY NPM_CONFIG_HTTP_PROXY NPM_CONFIG_HTTPS_PROXY
}

echo "started_at: $(date -Is)"
echo "slurm_job_id: ${SLURM_JOB_ID}"
echo "repo_root: ${REPO_ROOT}"
echo "cosmos_root: ${COSMOS_ROOT}"
echo "cosmos_commit: $(git -C "${COSMOS_ROOT}" rev-parse HEAD)"
echo "framework_root: ${FRAMEWORK_ROOT}"
echo "framework_commit: $(git -C "${FRAMEWORK_ROOT}" rev-parse HEAD)"
echo "run_root: ${RUN_ROOT}"
echo "num_chunks: ${NUM_CHUNKS}"
echo "num_gpus: ${NUM_GPUS}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: missing Cosmos Framework venv python: ${PYTHON_BIN}" >&2
  echo "Run scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh first." >&2
  exit 3
fi

if [[ -z "${COSMOS3_CHECKPOINT_PATH:-}" ]]; then
  echo "ERROR: set COSMOS3_CHECKPOINT_PATH to the official Cosmos3 checkpoint path before sbatch." >&2
  exit 4
fi

source "${FRAMEWORK_ROOT}/.venv/bin/activate"
unset_proxy_env
export PATH="${COSMOS_ROOT}/.tools:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${COSMOS_ROOT}/.uv_cache}"
export HF_HOME="${HF_HOME:-${COSMOS_ROOT}/checkpoints/hf_home}"
runtime_env="$("${PYTHON_BIN}" - "${FRAMEWORK_ROOT}" "${RUN_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

framework_root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
python_bin = framework_root / ".venv" / "bin" / "python"
site_packages = None
for candidate in sorted((python_bin.parent.parent / "lib").glob("python*/site-packages")):
    if (candidate / "nvidia").is_dir():
        site_packages = candidate
        break

paths = []
if site_packages:
    nvidia_root = site_packages / "nvidia"
    for lib_dir in sorted(nvidia_root.glob("**/lib")):
        if any(lib_dir.glob("lib*.so*")):
            paths.append(lib_dir)

    package_dir = site_packages / "nvidia" / "cuda_runtime"
    alias_dir = site_packages / "nvidia" / "cudart"
    if package_dir.is_dir():
        if alias_dir.is_symlink() and not alias_dir.exists():
            alias_dir.unlink()
        if not alias_dir.exists():
            alias_dir.symlink_to(package_dir, target_is_directory=True)

    for env_name, package_name in [
        ("CUDNN_HOME", "cudnn"),
        ("CUDART_HOME", "cuda_runtime"),
        ("NVRTC_HOME", "cuda_nvrtc"),
        ("CURAND_HOME", "curand"),
    ]:
        package_dir = site_packages / "nvidia" / package_name
        if package_dir.is_dir():
            print(f"export {env_name}={package_dir}")

    include_dir = site_packages / "nvidia" / "cuda_runtime" / "include"
    if include_dir.is_dir():
        print(f"export NVTE_CUDA_INCLUDE_DIR={include_dir}")

    av_libs = site_packages / "av.libs"
    if av_libs.is_dir():
        link_dir = run_root / "torchcodec_ffmpeg_links"
        soname_patterns = {
            "libavcodec.so.62": "libavcodec-*.so.62*",
            "libavdevice.so.62": "libavdevice-*.so.62*",
            "libavfilter.so.11": "libavfilter-*.so.11*",
            "libavformat.so.62": "libavformat-*.so.62*",
            "libavutil.so.60": "libavutil-*.so.60*",
            "libswresample.so.6": "libswresample-*.so.6*",
            "libswscale.so.9": "libswscale-*.so.9*",
        }
        linked_any = False
        for soname, pattern in soname_patterns.items():
            matches = sorted(av_libs.glob(pattern))
            if not matches:
                continue
            link_dir.mkdir(parents=True, exist_ok=True)
            link = link_dir / soname
            target = matches[-1].resolve()
            if link.exists() or link.is_symlink():
                if link.is_symlink() and link.resolve() == target:
                    linked_any = True
                    continue
                link.unlink()
            link.symlink_to(target)
            linked_any = True
        if linked_any:
            paths = [link_dir, av_libs, *paths]

if paths:
    merged = []
    for path in [*paths, *[Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]]:
        text = str(path)
        if text and text not in merged:
            merged.append(text)
    print("export LD_LIBRARY_PATH=" + ":".join(merged))
print(f"export PYTHONPATH={framework_root}:{os.environ.get('PYTHONPATH', '')}")
PY
)"
eval "${runtime_env}"
echo "runtime_env:"
echo "${runtime_env}"
FFMPEG_BIN="$("${PYTHON_BIN}" - <<'PY'
import imageio_ffmpeg

print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
echo "ffmpeg_bin: ${FFMPEG_BIN}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${COSMOS3_NANO_TEXT_MASTER_PORT:-29500}"
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COSMOS3_REPO="${FRAMEWORK_ROOT}"
export COSMOS3_INPUT_DIR="${INPUT_DIR}"
export COSMOS3_OUTPUT_ROOT="${RUN_ROOT}"
export COSMOS3_NUM_DROID_CHUNKS="${NUM_CHUNKS}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

from PIL import Image

cosmos_root = Path(os.environ["COSMOS3_REPO"]).parents[1]
framework_root = Path(os.environ["COSMOS3_REPO"])
input_dir = Path(os.environ["COSMOS3_INPUT_DIR"])
output_root = Path(os.environ["COSMOS3_OUTPUT_ROOT"])
num_chunks = int(os.environ["COSMOS3_NUM_DROID_CHUNKS"])
chunk_length = 16

if str(framework_root) not in sys.path:
    sys.path.insert(0, str(framework_root))

from cosmos_framework.data.vfm.action.datasets import DROIDLeRobotDataset

dataset_root = cosmos_root / "cookbooks/cosmos3/generator/action/assets/droid_lerobot_example"
dataset = DROIDLeRobotDataset(root=dataset_root)
chunk_starts = [idx * chunk_length for idx in range(num_chunks)]
if chunk_starts[-1] >= len(dataset):
    raise RuntimeError(f"requested chunk start {chunk_starts[-1]} but dataset length is {len(dataset)}")

input_dir.mkdir(parents=True, exist_ok=True)
initial_vision_path = input_dir / "robotics_droid_autoregressive_input_chunk_00.png"
records = []

for chunk_idx, sample_idx in enumerate(chunk_starts):
    sample = dataset[sample_idx]
    if int(sample["action"].shape[0]) != chunk_length:
        raise RuntimeError(f"expected {chunk_length} actions, got {sample['action'].shape[0]}")

    action_path = input_dir / f"robotics_droid_action_chunk_{chunk_idx:02d}.json"
    action_path.write_text(json.dumps(sample["action"].cpu().tolist()) + "\n", encoding="utf-8")

    if chunk_idx == 0:
        first_frame = sample["video"][:, 0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(first_frame).save(initial_vision_path)
        vision_path = initial_vision_path
    else:
        vision_path = input_dir / f"robotics_droid_autoregressive_input_chunk_{chunk_idx:02d}.png"

    records.append(
        {
            "action_chunk_size": chunk_length,
            "action_path": str(action_path),
            "domain_name": "droid_lerobot",
            "fps": int(sample["conditioning_fps"]),
            "image_size": 480,
            "view_point": sample["viewpoint"],
            "model_mode": "forward_dynamics",
            "name": f"robotics_action_cond_chunk_{chunk_idx:02d}",
            "prompt": sample["ai_caption"],
            "seed": 0,
            "vision_path": str(vision_path),
        }
    )

plan_path = input_dir / "action_forward_dynamics_robotics_custom.jsonl"
plan_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
(input_dir / "records.tsv").write_text(
    "\n".join(
        f"{idx}\t{record['name']}\t{record['seed']}\t{record['action_chunk_size']}\t{record['fps']}"
        for idx, record in enumerate(records)
    )
    + "\n",
    encoding="utf-8",
)

print("loaded DROID samples from:", dataset_root)
print("wrote initial conditioning image:", initial_vision_path)
print("wrote robotics plan:", plan_path)
print("output root:", output_root)
PY

while IFS=$'\t' read -r chunk_idx chunk_name seed action_chunk_size fps; do
  chunk_input="${INPUT_DIR}/action_forward_dynamics_robotics_chunk_$(printf '%02d' "${chunk_idx}").jsonl"
  "${PYTHON_BIN}" - "${INPUT_DIR}/action_forward_dynamics_robotics_custom.jsonl" "${chunk_idx}" "${chunk_input}" <<'PY'
import json
import sys
from pathlib import Path

plan = Path(sys.argv[1])
chunk_idx = int(sys.argv[2])
chunk_input = Path(sys.argv[3])
records = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines() if line.strip()]
chunk_input.write_text(json.dumps(records[chunk_idx]) + "\n", encoding="utf-8")
PY

  echo "running official DROID FD chunk ${chunk_idx}: ${chunk_name}"
  "${PYTHON_BIN}" -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    --no-guardrails \
    -i "${chunk_input}" \
    -o "${OUTPUT_DIR}" \
    --checkpoint-path "${COSMOS3_CHECKPOINT_PATH}" \
    --video-save-quality 8 \
    --image_size 480 \
    --seed "${seed}" \
    --benchmark

  output_video="${OUTPUT_DIR}/${chunk_name}/vision.mp4"
  if [[ ! -f "${output_video}" ]]; then
    echo "ERROR: missing generated video: ${output_video}" >&2
    exit 5
  fi
  echo "generated_video: ${output_video}"

  next_chunk=$((chunk_idx + 1))
  if [[ "${next_chunk}" -lt "${NUM_CHUNKS}" ]]; then
    next_vision="${INPUT_DIR}/robotics_droid_autoregressive_input_chunk_$(printf '%02d' "${next_chunk}").png"
    "${FFMPEG_BIN}" -y -loglevel error -i "${output_video}" \
      -vf "select=eq(n\\,${action_chunk_size})" \
      -frames:v 1 "${next_vision}"
    test -f "${next_vision}"
    "${PYTHON_BIN}" - "${INPUT_DIR}/action_forward_dynamics_robotics_custom.jsonl" "${next_chunk}" "${next_vision}" <<'PY'
import json
import sys
from pathlib import Path

plan = Path(sys.argv[1])
chunk_idx = int(sys.argv[2])
next_vision = sys.argv[3]
records = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines() if line.strip()]
records[chunk_idx]["vision_path"] = next_vision
plan.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
PY
  fi
done < "${INPUT_DIR}/records.tsv"

if find "${OUTPUT_DIR}" -name 'vision.mp4' -type f | grep -q .; then
  echo "status: success"
  find "${OUTPUT_DIR}" -name 'vision.mp4' -type f | sort > "${RUN_ROOT}/generated_videos.txt"
else
  echo "status: failed_no_video" >&2
  exit 6
fi

STITCH_DIR="${OUTPUT_DIR}/_stitched_segments"
mkdir -p "${STITCH_DIR}"
CONCAT_FILE="${STITCH_DIR}/concat.txt"
: > "${CONCAT_FILE}"

while IFS=$'\t' read -r chunk_idx chunk_name seed action_chunk_size fps; do
  src="${OUTPUT_DIR}/${chunk_name}/vision.mp4"
  segment="${STITCH_DIR}/${chunk_name}_generated.mp4"
  if [[ ! -f "${src}" ]]; then
    echo "ERROR: missing chunk video for stitch: ${src}" >&2
    exit 7
  fi
  "${FFMPEG_BIN}" -y -loglevel error -i "${src}" \
    -vf "select=gte(n\\,1),setpts=N/FRAME_RATE/TB" \
    -an \
    -r "${fps}" \
    -c:v libx264 \
    -crf 18 \
    -preset veryfast \
    -pix_fmt yuv420p \
    "${segment}"
  printf "file '%s'\n" "${segment}" >> "${CONCAT_FILE}"
done < "${INPUT_DIR}/records.tsv"

STITCHED_VIDEO="${OUTPUT_DIR}/robotics_action_cond_stitched.mp4"
"${FFMPEG_BIN}" -y -loglevel error \
  -f concat \
  -safe 0 \
  -i "${CONCAT_FILE}" \
  -c copy \
  "${STITCHED_VIDEO}"

if [[ ! -f "${STITCHED_VIDEO}" ]]; then
  echo "ERROR: missing stitched robotics video: ${STITCHED_VIDEO}" >&2
  exit 8
fi
echo "stitched_robotics_video: ${STITCHED_VIDEO}"
echo "${STITCHED_VIDEO}" > "${RUN_ROOT}/stitched_video.txt"

echo "finished_at: $(date -Is)"
echo "generated_videos_manifest: ${RUN_ROOT}/generated_videos.txt"
echo "stitched_video_manifest: ${RUN_ROOT}/stitched_video.txt"
