# Cosmos3 DROID Forward Dynamics Route A

## 目标

Route A 只验证 NVIDIA 官方 Cosmos3 DROID/LeRobot Forward Dynamics cookbook 是否能在本仓库环境中被定位、准备、提交并产出视频。当前阶段不训练、不接 Piper14、不做真机部署，也不修改现有 dataset、baseline policy、training 或 safety 代码。

官方 Cosmos 源码隔离在 `external/cosmos`。本仓库只新增轻量 wrapper、文档和报告。

## 官方依据

- `external/cosmos/README.md`：Cosmos3 是 omnimodal world model；Generator 支持 action modeling；Forward dynamics 的输入是 image + action chunk，输出是 video；action 参数包括 `action_mode`、`domain_name`、`raw_action_dim`、`action_chunk_size`、`action_path`。
- `external/cosmos/cookbooks/cosmos3/README.md`：官方 cookbook 环境要求 Linux、NVIDIA GPU、`uv`、`git`、`git-lfs`、HF token；Cosmos Framework 独立安装在 `packages/cosmos3/.venv`，使用 `uv sync --all-extras --group=<cuXXX-train>`。
- `external/cosmos/cookbooks/cosmos3/generator/action/README.md`：Action cookbook 入口。
- `external/cosmos/cookbooks/cosmos3/generator/action/run_fd_with_cosmos_framework.ipynb`：官方 DROID/LeRobot Forward Dynamics 示例，使用 `DROIDLeRobotDataset`、`domain_name="droid_lerobot"`、`model_mode="forward_dynamics"`、16-action chunk、`python -m cosmos_framework.scripts.inference`，输出检查 `vision.mp4`。
- 本仓库生成的证据文件：`reports/cosmos3_fd/official_fd_references.txt`。

当前官方 Cosmos commit：

```text
60b94f685cbe8f5e0ef4209be79514d07db1566f
```

## Forward Dynamics 定义

按官方 README，Forward Dynamics 是 action mode 的一种：模型用初始视觉上下文和动作序列预测未来视觉观测。对 DROID 例子，本任务只关心官方 cookbook 中的最小闭环：

- 输入：DROID/LeRobot 示例数据的初始图像 `vision_path` 和动作 chunk `action_path`。
- 条件字段：`domain_name="droid_lerobot"`、`action_chunk_size=16`、`model_mode="forward_dynamics"`。
- 输出：每个 chunk 的生成视频 `vision.mp4`。

## 新增脚本

### 1. 轻量官方检查

```bash
python scripts/cosmos3_fd/inspect_official_fd_cookbook.py
```

作用：

- 检查 `external/cosmos` 是否存在。
- 打印并记录官方 Cosmos commit。
- 定位官方 FD cookbook。
- 搜索并记录关键词：`forward_dynamics`、`droid_lerobot`、`action_chunk_size`、`vision_path`、`action_path`、`domain_name`、`model_mode`。
- 输出到 `reports/cosmos3_fd/official_fd_references.txt`。
- 不加载模型，不使用 GPU。

### 2. 准备官方环境

```bash
COSMOS3_BOOTSTRAP_UV=1 bash scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh
```

作用：

- 如果缺失，clone `https://github.com/NVIDIA/cosmos.git` 到 `external/cosmos`。
- 如果缺失，clone `https://github.com/NVIDIA/cosmos-framework.git` 到 `external/cosmos/packages/cosmos3`。
- 如果集群上有 conda，默认创建/复用 `external/cosmos/conda_envs/cosmos3` 作为外层 shell 环境；可用 `COSMOS3_CONDA_ENV_NAME` 或 `COSMOS3_CONDA_ENV_PREFIX` 覆盖。
- 如果 `uv` 不在 PATH 上，`COSMOS3_BOOTSTRAP_UV=1` 会把 `uv` 安装到 `external/cosmos/.tools`，不写 base/fastwam。
- 在 Cosmos Framework checkout 内运行官方 `uv sync --all-extras --group=${COSMOS3_UV_GROUP:-cu130-train}`。
- 环境只写入 `external/cosmos/packages/cosmos3/.venv`，不污染 base/fastwam。
- 下载步骤会清掉 `http_proxy`、`https_proxy`、`all_proxy` 等代理变量，走无代理通道。

CUDA 12.x 机器可用：

```bash
COSMOS3_BOOTSTRAP_UV=1 COSMOS3_UV_GROUP=cu128-train bash scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh
```

本集群 H800 节点的 `nvidia-smi` 显示 CUDA 12.8；Route A 当前应使用 `COSMOS3_UV_GROUP=cu128-train`。

## 下载官方 checkpoint

checkpoint 可以放在仓库隔离目录，不混入业务代码：

```bash
export HF_TOKEN=<your_hf_token>
COSMOS3_MODEL_ID=nvidia/Cosmos3-Nano bash scripts/cosmos3_fd/download_official_checkpoint.sh
export COSMOS3_CHECKPOINT_PATH="$PWD/external/cosmos/checkpoints/Cosmos3-Nano"
```

该脚本使用官方 `hf download`，输出到 `external/cosmos/checkpoints/<model-name>/`，并在下载前清掉代理变量。

## 探测 H800 GPU/CUDA 信息

先申请 1 块 GPU 做轻量探测，不加载模型：

```bash
export COSMOS3_SLURM_ACCOUNT=<your_PI_account>
bash scripts/cosmos3_fd/submit_gpu_probe.sh
```

## SLURM 运行 DROID FD

只能在计算节点/SLURM job 中跑。不要在 login 节点直接运行 inference。

先设置官方 checkpoint 路径：

```bash
export COSMOS3_CHECKPOINT_PATH=/path/to/official/cosmos3/checkpoint
```

提交最小 1 个 DROID chunk：

```bash
COSMOS3_SLURM_ACCOUNT=<your_PI_account> \
COSMOS3_NUM_GPUS=1 \
bash scripts/cosmos3_fd/submit_official_droid_fd.sh
```

如需按官方 notebook 跑 5 个连续 autoregressive chunks：

```bash
COSMOS3_SLURM_ACCOUNT=<your_PI_account> \
COSMOS3_NUM_GPUS=4 \
COSMOS3_DROID_NUM_CHUNKS=5 \
bash scripts/cosmos3_fd/submit_official_droid_fd.sh
```

输出位置：

```text
reports/cosmos3_fd/outputs/<timestamp>/
```

成功证据：

- `reports/cosmos3_fd/outputs/<timestamp>/run.log` 记录完整命令和状态。
- `reports/cosmos3_fd/outputs/<timestamp>/generated_videos.txt` 列出生成视频。
- 至少存在一个 `vision.mp4`。
- 多 chunk 运行会按官方 notebook 的 stitch 步骤生成 `action_forward_dynamics_robotics_custom/robotics_action_cond_stitched.mp4`。
- `reports/cosmos3_fd/outputs/<timestamp>/stitched_video.txt` 记录 stitched 视频路径。

本次 Route A 已跑通的输出：

```text
reports/cosmos3_fd/outputs/20260627_154359/action_forward_dynamics_robotics_custom/robotics_action_cond_chunk_00/vision.mp4
```

5 个 DROID chunks 跑完后的预期输出包括：

```text
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/robotics_action_cond_chunk_00/vision.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/robotics_action_cond_chunk_01/vision.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/robotics_action_cond_chunk_02/vision.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/robotics_action_cond_chunk_03/vision.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/robotics_action_cond_chunk_04/vision.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/_stitched_segments/robotics_action_cond_chunk_00_generated.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/_stitched_segments/robotics_action_cond_chunk_01_generated.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/_stitched_segments/robotics_action_cond_chunk_02_generated.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/_stitched_segments/robotics_action_cond_chunk_03_generated.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/_stitched_segments/robotics_action_cond_chunk_04_generated.mp4
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/_stitched_segments/concat.txt
reports/cosmos3_fd/outputs/<timestamp>/action_forward_dynamics_robotics_custom/robotics_action_cond_stitched.mp4
reports/cosmos3_fd/outputs/<timestamp>/stitched_video.txt
```

## 本任务明确不做

- 不训练 Cosmos 或 Piper14 模型。
- 不做 Piper14 adapter。
- 不做 DROID 到 Piper14 action/schema 映射。
- 不做真机部署或控制。
- 不修改 `piper_cosmos/`、`training/`、dataset、baseline policy、安全验证代码。
- 不在 login 节点加载 checkpoint 或运行 FD inference。
