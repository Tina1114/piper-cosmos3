# 仓库指南

## 1. 仓库定位

这个仓库不是通用机器人框架，而是一个围绕 `Cosmos3` 与 `Piper14 battery_assemble` 数据进行实验、训练和验证的工作仓库。

当前有两条主要工作线：

- `Piper14 SFT 主线`：把本地 `battery_assemble` 数据接到 Cosmos3，完成 DCP 转换、训练前检查、SLURM 提交训练、离线评估和安全检查。
- `Cosmos3 FD 验证支线`：验证 NVIDIA 官方 DROID/LeRobot Forward Dynamics cookbook 能否在本仓库环境中跑通。

如果你是第一次接手这个仓库，优先关注 `Piper14 SFT 主线`。如果你只想确认官方 Cosmos3 Forward Dynamics 是否能跑，再看 `scripts/cosmos3_fd/`。

## 2. 目录速览

### `piper_cosmos/`

仓库内的核心 Python 代码。

- `piper_cosmos/cosmos3/`：Cosmos3 接入层。
  负责注册 Piper14 domain、接入本地 HF 资产、组织训练入口、构造适配的数据集。
- `piper_cosmos/data/`：HDF5 数据读取与 split 工具。
- `piper_cosmos/safety/`：离线安全检查逻辑。
- `piper_cosmos/models/`：历史模型尝试和适配代码。这里不是当前真实 Cosmos3 SFT 的主入口。

### `scripts/`

主要操作入口。大多数实际流程都从这里开始。

- `submit_convert_cosmos3_nano_dcp.sh`：提交 DCP 转换任务。
- `cosmos3_piper14_readiness.py`：训练前 readiness 汇总检查。
- `preflight_cosmos3_piper14_sft.py`：训练前单项检查。
- `submit_cosmos3_piper14_sft.sh`：提交 Piper14 SFT。
- `cosmos3_piper14_offline_eval.py`：离线评估与预测导出。
- `verify_cosmos3_dcp.py`：检查 DCP 是否完整。
- `scripts/cosmos3_fd/`：官方 Forward Dynamics 路线相关脚本。

### `configs/`

训练和数据配置。

- `configs/cosmos3/sft/action_policy_piper14_nano.toml`：当前默认 Cosmos3 Piper14 SFT 配置。
- `configs/dataset_configs/battery_assemble_hdf5.yaml`：Battery HDF5 数据定义。
- `configs/safety/battery_piper14_safety.yaml`：安全检查配置。

### `docs/`

长期文档。

- 本文件：仓库总览和主流程。
- `ACTION_SCHEMA.md`：14D 动作定义。
- `DATA_SCHEMA.md`：数据字段说明。
- `cosmos3_battery_assemble_training_plan.md`：Piper14 SFT 的专题说明。
- `cosmos3_fd_routeA.md`：官方 FD Route A 说明。

### `reports/`

运行产物、检查结果和阶段报告。这里是输出区，不应当当作稳定源码区。

### `external/cosmos/`

官方 Cosmos / Cosmos Framework checkout、独立环境、checkpoint 和缓存。

## 3. 当前主线流程

当前推荐的日常流程是：

1. 准备 `external/cosmos/` 环境和本地资产。
2. 将本地 `Cosmos3-Nano` Hugging Face 权重转换为 DCP。
3. 跑 readiness / preflight 检查。
4. 通过 SLURM 提交 Piper14 SFT。
5. 训练后做离线评估和安全检查。

下面按顺序说明。

## 4. 环境与依赖约定

这个仓库默认遵循以下约定：

- 不修改 FastWAM 现有环境。
- 训练和 Cosmos3 相关依赖默认放在 `external/cosmos/packages/cosmos3/.venv`。
- Hugging Face 资产、checkpoint、缓存和 Cosmos 官方 checkout 放在 `external/cosmos/`。
- 较重任务走 SLURM，不在 login 节点直接硬跑训练或正式推理。

如果你发现某个流程要求安装包到 base 环境或 FastWAM 环境，优先判断它是否偏离了仓库当前约定。

## 5. Piper14 SFT 主线

### 5.1 数据与动作前提

这条主线默认使用：

- 数据根目录：`/project/peilab/wam/physical_WM/data/battery_assemble/perfect`
- 图像：三路 RGB，相机键来自 HDF5
- 状态：`qpos`
- 动作：`14D` 绝对关节目标

动作语义不要自行改写。需要先接受两个前提：

- 动作定义以 [ACTION_SCHEMA.md](/project/peilab/wam/cosmos3_cy/docs/ACTION_SCHEMA.md) 为准。
- Battery 数据字段与布局以 [battery_assemble_hdf5.yaml](/project/peilab/wam/cosmos3_cy/configs/dataset_configs/battery_assemble_hdf5.yaml) 为准；历史文档可作为辅助参考，但应以当前配置和代码实现为最终依据。

### 5.2 准备本地 Cosmos3 环境

训练入口默认依赖 `external/cosmos/packages/cosmos3/.venv/bin/python`。

如果官方 Cosmos 环境还没准备好，优先参考：

- [cosmos3_battery_assemble_training_plan.md](/project/peilab/wam/cosmos3_cy/docs/cosmos3_battery_assemble_training_plan.md)
- [cosmos3_fd_routeA.md](/project/peilab/wam/cosmos3_cy/docs/cosmos3_fd_routeA.md)

这个仓库本身不提供一个统一的“一键完整初始化脚本”；它更像是在现有 `external/cosmos/` 基础上追加 Piper14 接入。

### 5.3 准备本地 HF 资产与 DCP

训练前通常先确认三个东西：

- 本地 `Cosmos3-Nano` checkpoint
- 本地 `Qwen` snapshot
- 本地 `Wan2.2_VAE.pth`

将本地 HF checkpoint 转成 DCP 的提交入口是：

```bash
BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_convert_cosmos3_nano_dcp.sh
```

如果只想在本地检查 DCP 是否完整：

```bash
python scripts/verify_cosmos3_dcp.py --path external/cosmos/checkpoints/Cosmos3-Nano-DCP
```

相关文件：

- [submit_convert_cosmos3_nano_dcp.sh](/project/peilab/wam/cosmos3_cy/scripts/submit_convert_cosmos3_nano_dcp.sh)
- [convert_cosmos3_nano_to_dcp_offline.py](/project/peilab/wam/cosmos3_cy/scripts/convert_cosmos3_nano_to_dcp_offline.py)
- [verify_cosmos3_dcp.py](/project/peilab/wam/cosmos3_cy/scripts/verify_cosmos3_dcp.py)

### 5.4 跑 readiness / preflight

在正式提交 SFT 前，先跑 readiness。

最常用入口：

```bash
python scripts/cosmos3_piper14_readiness.py --require-slurm-account
```

它会汇总检查：

- Python 解释器是否为 Cosmos3 独立环境
- TOML 配置是否存在
- Battery 数据是否存在
- 数据配置是否存在
- DCP checkpoint 是否完整
- Wan VAE 是否存在
- 输出目录是否可写
- `BATTERY_SLURM_ACCOUNT` 是否可用
- W&B 相关鉴权前提是否满足当前模式

更底层的单项检查逻辑在：

- [cosmos3_piper14_readiness.py](/project/peilab/wam/cosmos3_cy/scripts/cosmos3_piper14_readiness.py)
- [preflight_cosmos3_piper14_sft.py](/project/peilab/wam/cosmos3_cy/scripts/preflight_cosmos3_piper14_sft.py)

默认报告会写到：

```text
reports/cosmos3_piper14/readiness.json
```

### 5.5 提交 Piper14 SFT

准备好 DCP 和 readiness 后，正式训练入口是：

```bash
BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_cosmos3_piper14_sft.sh
```

默认训练配置：

- TOML：`configs/cosmos3/sft/action_policy_piper14_nano.toml`
- Python：`external/cosmos/packages/cosmos3/.venv/bin/python`
- 输出目录：`reports/cosmos3_piper14`

如果你需要修改训练规模，优先改环境变量或独立配置文件，不要直接把临时实验参数写死到主配置里。

相关文件：

- [submit_cosmos3_piper14_sft.sh](/project/peilab/wam/cosmos3_cy/scripts/submit_cosmos3_piper14_sft.sh)
- [run_cosmos3_piper14_sft_slurm.sh](/project/peilab/wam/cosmos3_cy/scripts/run_cosmos3_piper14_sft_slurm.sh)
- [action_policy_piper14_nano.toml](/project/peilab/wam/cosmos3_cy/configs/cosmos3/sft/action_policy_piper14_nano.toml)
- [train_action_policy_piper14.py](/project/peilab/wam/cosmos3_cy/piper_cosmos/cosmos3/train_action_policy_piper14.py)

### 5.6 训练后离线评估

离线评估入口：

```bash
python scripts/cosmos3_piper14_offline_eval.py \
  --checkpoint <checkpoint> \
  --run-config <config.yaml>
```

这个脚本会导出：

- `pred_action`
- `current_qpos`
- `ground_truth_action`

用于后续 acceptance、安全分析或离线检查。

相关文件：

- [cosmos3_piper14_offline_eval.py](/project/peilab/wam/cosmos3_cy/scripts/cosmos3_piper14_offline_eval.py)
- [battery_piper14_safety.yaml](/project/peilab/wam/cosmos3_cy/configs/safety/battery_piper14_safety.yaml)
- [safety_filter.py](/project/peilab/wam/cosmos3_cy/piper_cosmos/safety/safety_filter.py)

## 6. Forward Dynamics 支线

如果你的目标不是训练，而是验证 NVIDIA 官方 DROID/LeRobot Forward Dynamics cookbook 是否能在当前环境跑通，使用 `scripts/cosmos3_fd/`。

常见流程：

1. 检查官方 cookbook 和关键引用是否存在。
2. 准备官方 Cosmos 环境。
3. 下载官方 checkpoint。
4. 申请 GPU 做 probe。
5. 提交官方 FD 推理任务。

主要入口：

```bash
python scripts/cosmos3_fd/inspect_official_fd_cookbook.py
```

```bash
COSMOS3_BOOTSTRAP_UV=1 COSMOS3_UV_GROUP=cu128-train \
bash scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh
```

```bash
COSMOS3_SLURM_ACCOUNT=<account> bash scripts/cosmos3_fd/submit_gpu_probe.sh
```

```bash
COSMOS3_SLURM_ACCOUNT=<account> bash scripts/cosmos3_fd/submit_official_droid_fd.sh
```

详细背景看：

- [cosmos3_fd_routeA.md](/project/peilab/wam/cosmos3_cy/docs/cosmos3_fd_routeA.md)

## 7. 代码结构说明

### `piper_cosmos/cosmos3/`

这是当前最重要的代码目录。

- `action_policy_piper14_nano.py`：Piper14 实验注册与接入定义。
- `domain.py`：Piper14 domain 相关定义。
- `piper14_hdf5_action_dataset.py`：把本地 HDF5 数据接成 Cosmos3 训练可用的数据集。
- `local_hf_assets.py`：本地 Qwen / Wan 等 HF 资产解析与环境引导。
- `train_action_policy_piper14.py`：先完成本地资产 bootstrap 和 Piper14 注册，再调用官方 Cosmos3 train 模块。

理解这几个文件，基本就能知道“本仓库是如何把 Piper14 接到 Cosmos3 上”的。

### `piper_cosmos/data/`

这一层负责通用数据辅助逻辑，例如：

- 扫描 HDF5 文件
- 构造 episode split
- 提供基础 reader

如果你需要新增数据集、换 split 规则，通常先看这里。

### `piper_cosmos/safety/`

这一层负责对预测动作做离线安全分析。默认思路是：

```text
当前 qpos + 目标 action -> 安全检查 -> 接受或回退到当前 qpos
```

它主要用于分析、验收和部署前约束，不是训练主入口。

### `piper_cosmos/models/`

这里当前没有承载真实 Cosmos3 SFT 主链路。

尤其是 [cosmos3_piper14_adapter.py](/project/peilab/wam/cosmos3_cy/piper_cosmos/models/cosmos3_piper14_adapter.py) 明确写着它是 `M5 dry-run adapter` 和 `engineering skeleton only`。它适合做 I/O 契约或 shape 验证，不应当误认为正式训练 backbone。

## 8. 常用文件索引

如果你只想快速定位文件，优先看下面这些：

- 主训练配置：
  [configs/cosmos3/sft/action_policy_piper14_nano.toml](/project/peilab/wam/cosmos3_cy/configs/cosmos3/sft/action_policy_piper14_nano.toml)
- 数据定义：
  [configs/dataset_configs/battery_assemble_hdf5.yaml](/project/peilab/wam/cosmos3_cy/configs/dataset_configs/battery_assemble_hdf5.yaml)
- 安全配置：
  [configs/safety/battery_piper14_safety.yaml](/project/peilab/wam/cosmos3_cy/configs/safety/battery_piper14_safety.yaml)
- DCP 转换：
  [scripts/submit_convert_cosmos3_nano_dcp.sh](/project/peilab/wam/cosmos3_cy/scripts/submit_convert_cosmos3_nano_dcp.sh)
- readiness：
  [scripts/cosmos3_piper14_readiness.py](/project/peilab/wam/cosmos3_cy/scripts/cosmos3_piper14_readiness.py)
- 训练提交：
  [scripts/submit_cosmos3_piper14_sft.sh](/project/peilab/wam/cosmos3_cy/scripts/submit_cosmos3_piper14_sft.sh)
- 离线评估：
  [scripts/cosmos3_piper14_offline_eval.py](/project/peilab/wam/cosmos3_cy/scripts/cosmos3_piper14_offline_eval.py)
- FD 支线：
  [scripts/cosmos3_fd](/project/peilab/wam/cosmos3_cy/scripts/cosmos3_fd)

## 9. 不该做的事

为了避免把仓库继续变乱，建议遵守下面这些边界：

- 不要把短期 smoke 结论写回长期总览文档。
- 不要把临时日志、缓存和一次性产物当成源码保存位置。
- 不要把 FastWAM 环境当成 Cosmos3 训练环境直接改。
- 不要把 `piper_cosmos/models/cosmos3_piper14_adapter.py` 当成当前正式训练主入口。
- 不要在 login 节点直接跑正式训练或重型推理。

## 10. 进一步阅读

需要更细的背景时，再看这些专题文档：

- [ACTION_SCHEMA.md](/project/peilab/wam/cosmos3_cy/docs/ACTION_SCHEMA.md)
- [DATA_SCHEMA.md](/project/peilab/wam/cosmos3_cy/docs/DATA_SCHEMA.md)
- [cosmos3_battery_assemble_training_plan.md](/project/peilab/wam/cosmos3_cy/docs/cosmos3_battery_assemble_training_plan.md)
- [cosmos3_fd_routeA.md](/project/peilab/wam/cosmos3_cy/docs/cosmos3_fd_routeA.md)
