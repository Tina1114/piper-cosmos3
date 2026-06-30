# piper-cosmos3

`piper-cosmos3` 是一个围绕 `Cosmos3` 与 `Piper14 battery_assemble` 数据开展训练、验证和离线检查的工作仓库，不是通用机器人框架。

当前仓库有两条主要工作线：

- `Piper14 SFT 主线`：将本地 `battery_assemble` 数据接入 Cosmos3，完成 DCP 转换、训练前检查、SLURM 提交训练、离线评估和安全检查。
- `Cosmos3 FD 验证支线`：验证 NVIDIA 官方 DROID/LeRobot Forward Dynamics cookbook 是否能在当前环境中跑通。

如果你是第一次接手这个仓库，优先看 `Piper14 SFT 主线`。

## 仓库结构

- `piper_cosmos/`：仓库内核心 Python 代码。
- `piper_cosmos/cosmos3/`：Cosmos3 接入层，包括 domain 注册、数据集适配、训练入口和本地资产接入。
- `piper_cosmos/data/`：HDF5 数据读取与 split 工具。
- `piper_cosmos/safety/`：离线安全检查逻辑。
- `scripts/`：主要操作入口，日常流程基本都从这里开始。
- `scripts/cosmos3_fd/`：官方 Forward Dynamics 支线脚本。
- `configs/`：训练、数据和安全配置。
- `docs/`：长期文档和专题说明。
- `reports/`：运行产物、检查结果和阶段输出，不是稳定源码区。
- `external/cosmos/`：官方 Cosmos checkout、独立环境、checkpoint 和外部依赖状态。

## 环境约定

这个仓库默认遵循以下约定：

- 不修改 FastWAM 现有环境。
- Cosmos3 相关依赖默认放在 `external/cosmos/packages/cosmos3/.venv`。
- 较重任务通过 SLURM 运行，不在 login 节点直接硬跑正式训练或推理。
- Hugging Face 资产、checkpoint、缓存和官方 Cosmos checkout 放在 `external/cosmos/`。

如果某个流程要求直接改 base 环境，先判断它是否偏离了当前仓库约定。

## Piper14 SFT 主线

这条主线默认使用：

- 数据根目录：`/project/peilab/wam/physical_WM/data/battery_assemble/perfect`
- 图像：三路 RGB，相机键来自 HDF5
- 状态：`qpos`
- 动作：`14D` 绝对关节目标

动作和数据字段的定义以这些文件为准：

- `docs/ACTION_SCHEMA.md`
- `docs/DATA_SCHEMA.md`
- `configs/dataset_configs/battery_assemble_hdf5.yaml`

### 常用流程

1. 准备 `external/cosmos/` 环境和本地 Hugging Face 资产。
2. 将本地 `Cosmos3-Nano` checkpoint 转成 DCP。
3. 运行 readiness / preflight 检查。
4. 通过 SLURM 提交 Piper14 SFT。
5. 训练后做离线评估和安全检查。

### 常用命令

检查训练前状态：

```bash
python scripts/cosmos3_piper14_readiness.py --require-slurm-account
```

提交 DCP 转换任务：

```bash
BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_convert_cosmos3_nano_dcp.sh
```

检查 DCP 是否完整：

```bash
python scripts/verify_cosmos3_dcp.py --path external/cosmos/checkpoints/Cosmos3-Nano-DCP
```

提交 Piper14 SFT：

```bash
BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_cosmos3_piper14_sft.sh
```

离线评估：

```bash
python scripts/cosmos3_piper14_offline_eval.py \
  --checkpoint <checkpoint> \
  --run-config <config.yaml>
```

当前默认核心配置：

- 训练 TOML：`configs/cosmos3/sft/action_policy_piper14_nano.toml`
- 数据配置：`configs/dataset_configs/battery_assemble_hdf5.yaml`
- 安全配置：`configs/safety/battery_piper14_safety.yaml`

## Cosmos3 FD 验证支线

如果你的目标不是训练，而是验证 NVIDIA 官方 DROID/LeRobot Forward Dynamics cookbook 是否能在当前环境跑通，使用 `scripts/cosmos3_fd/`。

常见流程：

1. 检查官方 cookbook 和关键引用是否存在。
2. 准备官方 Cosmos 环境。
3. 下载官方 checkpoint。
4. 申请 GPU 做 probe。
5. 提交官方 FD 推理任务。

常用入口：

```bash
python scripts/cosmos3_fd/inspect_official_fd_cookbook.py
```

```bash
COSMOS3_BOOTSTRAP_UV=1 COSMOS3_UV_GROUP=cu128-train \
bash scripts/cosmos3_fd/prepare_cosmos3_fd_env.sh
```

## 进一步阅读

- `docs/repo_guide.md`：仓库总览和主流程。
- `docs/cosmos3_battery_assemble_training_plan.md`：Piper14 SFT 专题说明。
- `docs/cosmos3_fd_routeA.md`：官方 FD Route A 说明。
- `docs/ACTION_SCHEMA.md`：14D 动作定义。
- `docs/DATA_SCHEMA.md`：数据字段说明。

## 注意事项

- `reports/` 主要是运行输出，不建议把它当作稳定源码目录使用。
- `external/cosmos/` 是外部依赖区，更新前先确认你要的是仓库源码变更，还是外部 checkout / 环境状态变更。
- 修改训练规模或实验参数时，优先使用独立配置或环境变量，不要把临时实验参数直接写死到主配置里。
