# Route A: Cosmos3 DROID Forward Dynamics 执行报告

## 基本信息

- 执行日期：2026-06-27
- 当前分支：`master`
- 目标：跑通 NVIDIA 官方 Cosmos3 DROID/LeRobot Forward Dynamics cookbook 路线，验证官方输入、输出、checkpoint load、H800 计算节点 CUDA 环境和生成视频流程。
- 结论：已跑通。SLURM job `460985` 在 `dgx-10` 上 `COMPLETED 0:0`，生成了 `vision.mp4`。
- 范围限制：未训练、未接 Piper14、未真机部署、未修改现有训练/dataset/baseline/safety 代码。

## 当前 git 状态

当前 worktree 在开始前已有大量未提交修改和未跟踪文件，主要涉及 docs、reports、`piper_cosmos/`、safety、baseline、M5 adapter 等。本次 Route A 只新增/修改以下路径：

- `scripts/cosmos3_fd/`
- `docs/cosmos3_fd_routeA.md`
- `docs/superpowers/specs/2026-06-27-cosmos3-fd-routea-design.md`
- `docs/superpowers/plans/2026-06-27-cosmos3-fd-routea.md`
- `reports/cosmos3_fd/`
- `tests/test_cosmos3_fd_routea.py`
- `external/cosmos/`

未修改现有训练、dataset、baseline policy、安全验证代码。

## 官方源码和依据

- 官方 Cosmos checkout：`external/cosmos`
- 官方 Cosmos commit：`60b94f685cbe8f5e0ef4209be79514d07db1566f`
- 官方 Cosmos Framework checkout：`external/cosmos/packages/cosmos3`
- 官方 Cosmos Framework commit：`90cd348877c37b888942c988b631eb1611bf2950`
- 官方 FD cookbook：`external/cosmos/cookbooks/cosmos3/generator/action/run_fd_with_cosmos_framework.ipynb`
- 官方 action cookbook README：`external/cosmos/cookbooks/cosmos3/generator/action/README.md`
- 官方环境 README：`external/cosmos/cookbooks/cosmos3/README.md`
- 官方总 README：`external/cosmos/README.md`
- 本地证据文件：`reports/cosmos3_fd/official_fd_references.txt`

证据文件已记录以下关键词位置：

- `forward_dynamics`
- `droid_lerobot`
- `action_chunk_size`
- `vision_path`
- `action_path`
- `domain_name`
- `model_mode`

## 环境和下载状态

- 独立 conda prefix：`external/cosmos/conda_envs/cosmos3`
- 隔离 `uv`：`external/cosmos/.tools/uv`
- Cosmos Framework venv：`external/cosmos/packages/cosmos3/.venv`
- 当前 CUDA wheel 组：`cu128-train`
- 当前 torch：`2.10.0+cu128`
- Hugging Face 用户：缓存 token 通过 `hf auth whoami` 验证为 `m1ku2`。
- 官方主 checkpoint 已下载：
  - repo：`nvidia/Cosmos3-Nano`
  - 路径：`external/cosmos/checkpoints/Cosmos3-Nano`
  - 下载结果：67/67 文件完成
- 推理过程中官方 framework 自动下载/使用了额外官方依赖到 `external/cosmos/checkpoints/hf_home`，包括：
  - `Qwen/Qwen3-VL-8B-Instruct`
  - `Wan-AI/Wan2.2-TI2V-5B` 的 `Wan2.2_VAE.pth`
  - `nvidia/Cosmos3-Nano` 的 `sound_tokenizer/*`

说明：所有 Cosmos 源码、环境、checkpoint 和 HF cache 均放在 `external/cosmos/` 隔离目录下，未安装到 base/fastwam。

## 已执行的关键命令

轻量检查：

```bash
python scripts/cosmos3_fd/inspect_official_fd_cookbook.py
python -m py_compile scripts/cosmos3_fd/inspect_official_fd_cookbook.py
bash -n scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh scripts/cosmos3_fd/run_official_droid_fd_slurm.sh scripts/cosmos3_fd/download_official_checkpoint.sh scripts/cosmos3_fd/probe_h800_gpu_slurm.sh scripts/cosmos3_fd/submit_gpu_probe.sh scripts/cosmos3_fd/submit_official_droid_fd.sh
python -m unittest tests.test_cosmos3_fd_routea -v
```

环境准备：

```bash
COSMOS3_BOOTSTRAP_UV=1 COSMOS3_UV_GROUP=cu128-train bash scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh
```

checkpoint 下载：

```bash
COSMOS3_MODEL_ID=nvidia/Cosmos3-Nano bash scripts/cosmos3_fd/download_official_checkpoint.sh
```

GPU probe：

```bash
COSMOS3_SLURM_ACCOUNT=peilab bash scripts/cosmos3_fd/submit_gpu_probe.sh
```

DROID FD 推理：

```bash
COSMOS3_SLURM_ACCOUNT=peilab \
COSMOS3_CHECKPOINT_PATH=/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Nano \
COSMOS3_NUM_GPUS=1 \
COSMOS3_DROID_NUM_CHUNKS=1 \
bash scripts/cosmos3_fd/submit_official_droid_fd.sh
```

## 轻量验证结果

- `python -m py_compile scripts/cosmos3_fd/inspect_official_fd_cookbook.py`：通过。
- `bash -n scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh scripts/cosmos3_fd/run_official_droid_fd_slurm.sh scripts/cosmos3_fd/download_official_checkpoint.sh scripts/cosmos3_fd/probe_h800_gpu_slurm.sh scripts/cosmos3_fd/submit_gpu_probe.sh scripts/cosmos3_fd/submit_official_droid_fd.sh`：通过。
- `python -m unittest tests.test_cosmos3_fd_routea -v`：通过。
- `python scripts/cosmos3_fd/inspect_official_fd_cookbook.py`：通过，已写入 `reports/cosmos3_fd/official_fd_references.txt`。
- `bash scripts/cosmos3_fd/run_official_droid_fd_slurm.sh`：按预期拒绝在非 SLURM 环境运行，未加载 checkpoint，未使用 GPU。

## GPU 验证结果

已申请 1 块 H800 做计算节点探测。最终成功作业：

- SLURM job：`460977`
- 节点：`dgx-21`
- 状态：`COMPLETED 0:0`
- GPU：`NVIDIA H800`
- 显存：约 81559 MiB
- Driver：`570.158.01`
- `nvidia-smi` CUDA Version：`12.8`
- torch：`2.10.0+cu128`
- `torch_cuda: 12.8`
- `cuda_available: True`
- `device_count: 1`
- `device_0_capability: (9, 0)`

失败历史：

- job `460971` 使用 `cu130-train`，在 H800 driver CUDA 12.8 环境下失败，torch 报 driver 太旧。已切换到 `cu128-train` 并通过 job `460977` 验证。

## DROID FD 推理结果

最终成功作业：

- SLURM job：`460985`
- 节点：`dgx-10`
- 状态：`COMPLETED 0:0`
- elapsed：`00:05:43`
- MaxRSS：`10609316K`
- 输出根目录：`reports/cosmos3_fd/outputs/20260627_154359`
- 日志：
  - `reports/cosmos3_fd/slurm_460985.out`
  - `reports/cosmos3_fd/slurm_460985.err`
  - `reports/cosmos3_fd/outputs/20260627_154359/run.log`
  - `reports/cosmos3_fd/outputs/20260627_154359/action_forward_dynamics_robotics_custom/console.log`
  - `reports/cosmos3_fd/outputs/20260627_154359/action_forward_dynamics_robotics_custom/debug.log`

生成视频：

```text
reports/cosmos3_fd/outputs/20260627_154359/action_forward_dynamics_robotics_custom/robotics_action_cond_chunk_00/vision.mp4
```

视频和产物证据：

- `vision.mp4` 文件存在，大小约 `453K`。
- `generated_videos.txt` 存在，内容指向上述 `vision.mp4`。
- `sample_outputs.json` 存在，包含 `"status":"success"`，并在 `outputs[0].files` 中列出上述 `vision.mp4`。
- `benchmark.json` 存在，记录：
  - `OmniInference.generate_batch`: 46.8468 s
  - `OmniMoTModel.generate_samples_from_batch`: 45.0015 s
  - `OmniMoTModel.decode`: 0.2943 s
- `slurm_460985.out` 包含：
  - `Saved sample outputs`
  - `Saved benchmark`
  - `generated_video: .../vision.mp4`
  - `status: success`
  - `generated_videos_manifest: .../generated_videos.txt`

推理输入符合 Route A 限制：

- 使用官方 DROID/LeRobot 示例：`external/cosmos/cookbooks/cosmos3/generator/action/assets/droid_lerobot_example`
- `domain_name`: `droid_lerobot`
- `model_mode`: `forward_dynamics`
- `action_chunk_size`: `16`
- `COSMOS3_DROID_NUM_CHUNKS=1`
- 未使用 Piper14 数据、未训练、未真机部署。

## 推理失败历史和修复

- job `460982`：失败于 torchcodec 加载，缺少 `libnppicc.so.12`。已在 runner 中补充 venv 内 NVIDIA wheel 的 runtime lib path、PyAV ffmpeg symlink、`CUDNN_HOME`、`CUDART_HOME`、`NVRTC_HOME`、`CURAND_HOME`、`NVTE_CUDA_INCLUDE_DIR` 和 `PYTHONPATH`。
- job `460984`：进入模型初始化后，官方 framework 内部 `uvx --with click hf@1.16.4 download ...` 下载 Qwen 时走代理/PyPI 失败。已在 runner 中清理 proxy 环境变量，并设置 `UV_CACHE_DIR`、`HF_HOME`。
- job `460985`：完成推理并生成视频。日志中仍出现一次 `/usr/bin/nvcc` 导入 `colorama` 失败的 traceback，但采样继续完成，SLURM 退出码为 `0:0`，官方 inference 日志写出 `SUCCESS`、`status: success` 和 `vision.mp4`。该 traceback 记录为非致命残留告警。

## 需要复现时的命令

准备环境：

```bash
COSMOS3_BOOTSTRAP_UV=1 COSMOS3_UV_GROUP=cu128-train bash scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh
```

下载官方 checkpoint：

```bash
COSMOS3_MODEL_ID=nvidia/Cosmos3-Nano bash scripts/cosmos3_fd/download_official_checkpoint.sh
```

提交 1 块 H800 的 GPU probe：

```bash
COSMOS3_SLURM_ACCOUNT=peilab bash scripts/cosmos3_fd/submit_gpu_probe.sh
```

提交最小 DROID FD inference：

```bash
COSMOS3_SLURM_ACCOUNT=peilab \
COSMOS3_CHECKPOINT_PATH="$PWD/external/cosmos/checkpoints/Cosmos3-Nano" \
COSMOS3_NUM_GPUS=1 \
COSMOS3_DROID_NUM_CHUNKS=1 \
bash scripts/cosmos3_fd/submit_official_droid_fd.sh
```

按官方 notebook 连续跑更多 DROID chunks 时可增加：

```bash
COSMOS3_DROID_NUM_CHUNKS=5
```

注意：当前 runner 会顺序处理 chunks；`COSMOS3_NUM_GPUS` 只影响 SLURM 申请数量，当前最小验证已用 1 块 H800 跑通。

## 最终状态

Route A 已达到目标：官方源码和 cookbook 已定位，隔离环境已建立，官方 checkpoint 已下载，H800 计算节点 CUDA 已验证，官方 DROID/LeRobot Forward Dynamics 最小 inference 已在 SLURM 中完成，并生成 `vision.mp4`。
