# Cosmos3 Piper14 推理环境配置

这份文档面向只拿到 `inference-only` 交付包的人，用来配置 Cosmos3 Piper14 battery policy 的推理环境。

目标是启动 policy server，加载导出的 checkpoint，并通过 RPC 返回 Piper14 action chunk。

## 1. 交付包结构

推荐把交付包整理成下面的结构：

```text
inference-only/
  checkpoints/
    12k/
    16k/
    18k/
    20k/
  provenance/
    config.yaml
    job_env.yaml
    launch_info.yaml
```

每个 checkpoint 目录必须包含：

```text
config.json
checkpoint.json
model.safetensors.index.json
model-00001-of-00007.safetensors
...
model-00007-of-00007.safetensors
```

推荐优先使用：

```text
inference-only/checkpoints/20k
```

## 2. 硬件要求

最低建议：

- NVIDIA GPU，显存建议 80 GB 级别。
- Linux GPU 节点。
- CUDA 12.8 运行栈。
- 推理服务器和客户端网络互通。

训练时使用的是 4 GPU 配置，但 inference-only checkpoint 已经导出为 Hugging Face/safetensors 形式；推理通常按单个 policy server 进程部署。

## 3. Python 环境

推荐使用和训练/导出一致的 CUDA 12.8 Python 环境：

```text
Python: 3.13.x
PyTorch: 2.10.0+cu128
Transformers: 4.57.x
CUDA group: cu128
```

环境中需要能 import：

```text
torch
transformers
safetensors
cosmos_framework
piper_cosmos
numpy
PIL
```

如果使用项目自带的 Cosmos Framework 环境，确保 `PYTHONPATH` 同时包含：

```bash
export PYTHONPATH="<repo-root>:<cosmos3-framework-root>:${PYTHONPATH}"
```

其中：

- `<repo-root>` 是包含 `piper_cosmos/` 和 `scripts/` 的代码仓库根目录。
- `<cosmos3-framework-root>` 是包含 `cosmos_framework/` 的 Cosmos3 framework 代码目录。

## 4. 离线模型资产

推理环境需要下列本地模型资产：

```text
Qwen/Qwen3-VL-8B-Instruct
Wan-AI/Wan2.2-TI2V-5B 中的 Wan2.2_VAE.pth
```

建议放在部署机器自己的模型缓存目录中，例如：

```text
<model-cache>/
  Qwen/Qwen3-VL-8B-Instruct/
  Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

然后设置环境变量：

```bash
export HF_HOME="<model-cache>"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS3_QWEN_SNAPSHOT="<model-cache>/Qwen/Qwen3-VL-8B-Instruct"
export COSMOS3_WAN_VAE_PATH="<model-cache>/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
```

不要在生产推理启动时依赖联网下载模型。

## 5. 运行时环境变量

最小配置示例：

```bash
export REPO_ROOT="<repo-root>"
export INFERENCE_ROOT="<inference-only-root>"
export CHECKPOINT_DIR="${INFERENCE_ROOT}/checkpoints/20k"
export HF_HOME="<model-cache>"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS3_QWEN_SNAPSHOT="<qwen3-vl-8b-instruct-snapshot>"
export COSMOS3_WAN_VAE_PATH="<wan2.2-vae-pth>"
export PYTHONPATH="${REPO_ROOT}:<cosmos3-framework-root>:${PYTHONPATH}"
```

可选缓存目录：

```bash
export XDG_CACHE_HOME="<runtime-cache>/xdg"
export XDG_DATA_HOME="<runtime-cache>/xdg"
export IMAGINAIRE_CACHE_DIR="<runtime-cache>/imaginaire"
export MPLCONFIGDIR="<runtime-cache>/matplotlib"
```

## 6. 启动 Policy Server

在 GPU 节点启动：

```bash
cd "${REPO_ROOT}"

python scripts/serve_cosmos_piper14_policy.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --host 0.0.0.0 \
  --port 8766
```

如果需要指定 Python：

```bash
<python-bin> scripts/serve_cosmos_piper14_policy.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --host 0.0.0.0 \
  --port 8766
```

## 7. 客户端连接

客户端只需要能访问 server 的 host 和 port：

```text
host: <gpu-node-hostname-or-ip>
port: 8766
```

server 预期返回：

```text
domain: piper14
action type: absolute_joint_position_command
action chunk shape: [32, 14]
image keys: cam_high, cam_left_wrist, cam_right_wrist
image format: 480x640 RGB uint8
```

## 8. 验收检查

部署后至少检查：

- `CHECKPOINT_DIR` 指向 `inference-only/checkpoints/20k` 或其他导出 checkpoint。
- checkpoint 目录中有 `config.json`、`checkpoint.json`、index JSON 和 7 个 safetensors shard。
- `torch.cuda.is_available()` 为 `True`。
- `torch.__version__` 是 CUDA 12.8 对应版本。
- `cosmos_framework` 和 `piper_cosmos` 可以 import。
- `COSMOS3_QWEN_SNAPSHOT` 指向本地 Qwen3-VL snapshot。
- `COSMOS3_WAN_VAE_PATH` 指向本地 `Wan2.2_VAE.pth`。
- `HF_HUB_OFFLINE=1`，`TRANSFORMERS_OFFLINE=1`。
- server 启动后 metadata 中 action chunk 为 `[32, 14]`。
- 返回 action 全部是 finite 数值，没有 NaN/Inf。

可以用下面的命令快速检查核心 Python 环境：

```bash
python - <<'PY'
import torch
import transformers
import cosmos_framework
import piper_cosmos

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("cosmos_framework: import ok")
print("piper_cosmos: import ok")
PY
```

## 9. 常见问题

### checkpoint 加载失败

优先确认使用的是导出的 inference-only checkpoint，而不是训练时的 DCP checkpoint 目录。

正确：

```text
inference-only/checkpoints/20k
```

错误：

```text
训练过程中的 raw DCP checkpoint
```

### 启动时尝试联网下载模型

检查：

```bash
echo "${HF_HUB_OFFLINE}"
echo "${TRANSFORMERS_OFFLINE}"
echo "${COSMOS3_QWEN_SNAPSHOT}"
echo "${COSMOS3_WAN_VAE_PATH}"
```

两项 offline 变量应为 `1`，两个模型路径应指向部署机器本地已有文件/目录。

### 找不到 Piper14 domain 或 action policy

确认 `PYTHONPATH` 包含 `<repo-root>`，并且仓库中存在：

```text
piper_cosmos/cosmos3/domain.py
piper_cosmos/cosmos3/action_policy_piper14_nano.py
```

### 推理速度不够 30 Hz

该 policy server 返回的是 `[32, 14]` action chunk。实际闭环控制建议使用 action buffer，以较低频率重规划，不要把单次 Cosmos 推理当作 30 Hz 同步控制步骤。

## 10. 一页配置摘要

```bash
export REPO_ROOT="<repo-root>"
export INFERENCE_ROOT="<inference-only-root>"
export CHECKPOINT_DIR="${INFERENCE_ROOT}/checkpoints/20k"

export HF_HOME="<model-cache>"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS3_QWEN_SNAPSHOT="<qwen3-vl-8b-instruct-snapshot>"
export COSMOS3_WAN_VAE_PATH="<wan2.2-vae-pth>"

export PYTHONPATH="${REPO_ROOT}:<cosmos3-framework-root>:${PYTHONPATH}"

cd "${REPO_ROOT}"
python scripts/serve_cosmos_piper14_policy.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --host 0.0.0.0 \
  --port 8766
```
