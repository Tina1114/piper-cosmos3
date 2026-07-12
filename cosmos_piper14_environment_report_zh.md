# Cosmos Piper14 20k 推理环境配置与验证报告

## 背景

`piper-cosmos3` 的 inference-only 分支只提供 Piper14 policy server 的部署封装，不包含完整 Cosmos3 推理框架。真实加载 20k 后训练权重进行推理时，需要本机同时具备：

- 独立 Python/conda 推理环境。
- NVIDIA `cosmos-framework` 代码与依赖。
- Piper14 20k checkpoint。
- Qwen3-VL tokenizer/config 本地缓存。
- Wan2.2 VAE 本地文件。
- 可用 NVIDIA GPU 与匹配的 PyTorch/CUDA wheel。

## 本机硬件与 CUDA 判断

本机实际情况：

- GPU: NVIDIA GeForce RTX 4090
- 显存: 24564 MiB，约 24 GB
- NVIDIA Driver: 580.95.05
- `nvidia-smi` CUDA Version: 13.0
- `/usr/local/cuda`: CUDA 11.3 toolkit

判断：

- PyTorch `cu128` wheel 可以运行，因为 PyTorch wheel 自带 CUDA runtime，主要依赖 NVIDIA driver 兼容性；本机 580 驱动满足 CUDA 12.8 wheel 运行要求。
- `/usr/local/cuda-11.3` 不是运行 PyTorch `cu128` wheel 的阻塞项，除非后续需要本机编译 CUDA 扩展。
- 但本机 RTX 4090 只有 24 GB 显存，低于文档建议的 80 GB 级显存，真实加载完整 Cosmos3 Piper14 20k policy 时存在显存不足风险。

## 新建独立 Conda 环境

按要求没有复用或 clone `fastwam`，而是新建了独立环境：

```text
/home/agilex/miniconda3/envs/cosmos
```

核心版本：

```text
Python 3.13.14
torch 2.10.0+cu128
torchvision 0.25.0+cu128
transformers 4.57.6
safetensors 0.8.0
numpy 2.5.1
Pillow 12.3.0
huggingface_hub 0.36.2
cosmos-framework 1.2.2
```

CUDA 验证结果：

```text
torch.cuda.is_available(): True
device: NVIDIA GeForce RTX 4090
torch CUDA build: 12.8
```

## 官方 Cosmos Framework

已下载官方仓库：

```text
https://github.com/NVIDIA/cosmos-framework.git
```

本地位置：

```text
/home/agilex/World_Action_Model/physical_WM/src/piper-cosmos3/external/cosmos
```

当前 commit：

```text
3017efb48826b69f9582ac6e3c001f50d1401067
```

说明：

- 主仓库中的 `external/cosmos` 是 gitlink/subproject 形式，只记录指向的 commit。
- 不会把整个 framework 源码内容作为普通文件展开提交到 `piper-cosmos3` 主仓库。

## 本地模型资产

Piper14 20k checkpoint：

```text
/home/agilex/World_Action_Model/physical_WM/checkpoints/cosmos_battery/20k
```

已确认包含：

```text
config.json
checkpoint.json
model.safetensors.index.json
model-00001-of-00007.safetensors
...
model-00007-of-00007.safetensors
```

Wan2.2 VAE 复用已有文件：

```text
/home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

Qwen3-VL tokenizer/config 缓存：

```text
/home/agilex/World_Action_Model/physical_WM/checkpoints/hf_home
```

只缓存推理需要的 tokenizer/config 小文件，没有保留 Qwen safetensors/bin 大权重。

## 本地配置副本

新增本地配置：

```text
configs/cosmos_piper14_20k_local_config.json
```

用途：

- 不修改 20k checkpoint 原始 `config.json`。
- 将 Wan VAE 路径改成本机已有 VAE 文件。
- 将 Qwen tokenizer 路径改成本机 HuggingFace snapshot 绝对路径，避免离线模式下 `Qwen/Qwen3-VL-8B-Instruct` 解析失败。

## 代码适配

官方当前 `cosmos-framework` 使用的 action 数据路径是：

```text
cosmos_framework.data.generator.*
```

而 `piper-cosmos3` 原始封装引用的是旧路径：

```text
cosmos_framework.data.vfm.*
```

因此对以下文件做了小范围 import 路径适配：

```text
piper_cosmos/cosmos3/domain.py
piper_cosmos/deployment/cosmos_piper14_policy.py
```

主要替换：

```text
cosmos_framework.data.vfm.action
-> cosmos_framework.data.generator.action

cosmos_framework.data.vfm.joint_dataloader
-> cosmos_framework.data.generator.joint_dataloader
```

## 新增脚本

新增环境脚本：

```text
scripts/env_cosmos_piper14_20k.sh
```

作用：

- 设置 `REPO_ROOT`
- 设置 `CHECKPOINT_DIR`
- 设置 `CONFIG_FILE`
- 设置 `HF_HOME`
- 设置离线模式 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`
- 设置 Qwen/Wan 本地路径
- 设置 runtime cache 和 `TMPDIR`
- 设置 `PYTHONPATH`
- 指定 `cosmos` 环境 Python

新增检查脚本：

```text
scripts/check_cosmos_piper14_20k_env.sh
```

作用：

- 检查 20k checkpoint 文件完整性。
- 检查 Qwen tokenizer/config。
- 检查 Wan2.2 VAE。
- 检查 Python 包导入。
- 检查 `cosmos_framework` 是否可见。
- 输出 GPU 名称和显存。

新增启动脚本：

```text
scripts/start_cosmos_piper14_20k_server.sh
```

作用：

- source 环境脚本。
- 使用本地 20k checkpoint 和 local config 启动 policy server。

## 验证结果

基础环境检查通过：

```text
torch: ok
transformers: ok
safetensors: ok
numpy: ok
PIL: ok
piper_cosmos: ok
cosmos_framework: ok
filesystem checks passed
```

深层 import 验证通过：

```text
cosmos_framework.inference.common.args.CheckpointOverrides
cosmos_framework.scripts.action_policy_server_libero.ActionModelService
cosmos_framework.data.generator.action.transforms.ActionTransformPipeline
cosmos_framework.data.generator.action.domain_utils.get_domain_id
piper14 domain id = 21
```

mock backend 验证通过：

```text
domain_name: piper14
raw_action_dim: 14
action_horizon: 32
image_keys: cam_high, cam_left_wrist, cam_right_wrist
action_type: absolute_joint_position_command
```

真实 backend 初始化测试结果：

- Qwen tokenizer 加载通过。
- Wan2.2 VAE 加载通过。
- Cosmos 模型开始构建。
- 最终在模型搬到 GPU 阶段失败，原因是 CUDA OOM。

OOM 信息摘要：

```text
torch.OutOfMemoryError: CUDA out of memory
GPU total: 23.51 GiB
process used: 22.78 GiB
free: 96.25 MiB
```

测试结束后 GPU 显存已释放：

```text
RTX 4090 total 24564 MiB, used 652 MiB, free 23427 MiB
```

## 结论

软件环境、依赖、官方 framework、本地模型缓存、Piper14 部署封装已经对齐。

当前机器可以完成：

- 环境检查
- import 检查
- mock backend 流程验证
- checkpoint/config/cache 路径验证

当前机器不能完成：

- 完整 20k Cosmos3 Piper14 真实推理

阻塞原因不是依赖缺失，而是本机 RTX 4090 24 GB 显存不足。真实 policy server 建议迁移到 80 GB 级 GPU 机器运行，或者后续专门做量化、CPU offload、模型裁剪等显存优化适配。
