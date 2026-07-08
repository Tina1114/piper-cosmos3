# Cosmos3 Piper14 Policy SFT 流程与完成记录

## 1. 文档目的

这份文档只保留两类内容：

- 长期可维护的 `policy SFT` 正式流程
- 一段简短的“本次已经完成了什么”记录

它不再承担早期大而全计划文档的角色，也不记录一次性 smoke 细节。

## 2. 当前目标

当前主线是把 `battery_assemble` 的 Piper14 真机数据接到 Cosmos3，完成：

```text
本地 HF 资产
-> Cosmos3-Nano DCP 转换
-> readiness / preflight
-> SLURM 提交 policy SFT
-> 训练后离线评估与安全检查
```

动作接口保持为 `14D` 绝对关节目标，不做 DROID 8D 适配，也不把它改写为 dual-arm 20D EEF 动作。

## 3. 正式入口

当前正式训练链路是：

```text
scripts/submit_cosmos3_piper14_sft.sh
-> scripts/run_cosmos3_piper14_sft_slurm.sh
-> piper_cosmos.cosmos3.train_action_policy_piper14
-> piper_cosmos.cosmos3.action_policy_piper14_nano
-> cosmos_framework.scripts.train
```

这条链路依赖的关键文件：

- [configs/cosmos3/sft/action_policy_piper14_nano.toml](/project/peilab/wam/cosmos3_cy/configs/cosmos3/sft/action_policy_piper14_nano.toml)
- [piper_cosmos/cosmos3/action_policy_piper14_nano.py](/project/peilab/wam/cosmos3_cy/piper_cosmos/cosmos3/action_policy_piper14_nano.py)
- [piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py](/project/peilab/wam/cosmos3_cy/piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py)
- [piper_cosmos/cosmos3/domain.py](/project/peilab/wam/cosmos3_cy/piper_cosmos/cosmos3/domain.py)
- [piper_cosmos/cosmos3/train_action_policy_piper14.py](/project/peilab/wam/cosmos3_cy/piper_cosmos/cosmos3/train_action_policy_piper14.py)

## 4. 前置条件

正式提交训练前，默认应满足以下条件：

- `external/cosmos/` 已存在并可用
- `external/cosmos/packages/cosmos3/.venv/bin/python` 可用
- 本地 `Cosmos3-Nano` checkpoint 已准备好
- 本地 `Qwen` snapshot 已准备好
- 本地 `Wan2.2_VAE.pth` 已准备好
- Battery 数据目录存在
- `BATTERY_SLURM_ACCOUNT` 已设置

默认数据和配置入口：

- 数据根目录：`/project/peilab/wam/physical_WM/data/battery_assemble/perfect`
- 数据配置：`configs/dataset_configs/battery_assemble_hdf5.yaml`
- 训练配置：`configs/cosmos3/sft/action_policy_piper14_nano.toml`

## 5. 正式流程

### 5.1 本地 HF 资产与 DCP

训练前先把本地 `Cosmos3-Nano` HF 权重转换为 DCP。

提交入口：

```bash
BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_convert_cosmos3_nano_dcp.sh
```

本地验证入口：

```bash
python scripts/verify_cosmos3_dcp.py --path external/cosmos/checkpoints/Cosmos3-Nano-DCP
```

相关文件：

- [scripts/submit_convert_cosmos3_nano_dcp.sh](/project/peilab/wam/cosmos3_cy/scripts/submit_convert_cosmos3_nano_dcp.sh)
- [scripts/run_convert_cosmos3_nano_dcp_slurm.sh](/project/peilab/wam/cosmos3_cy/scripts/run_convert_cosmos3_nano_dcp_slurm.sh)
- [scripts/convert_cosmos3_nano_to_dcp_offline.py](/project/peilab/wam/cosmos3_cy/scripts/convert_cosmos3_nano_to_dcp_offline.py)
- [scripts/verify_cosmos3_dcp.py](/project/peilab/wam/cosmos3_cy/scripts/verify_cosmos3_dcp.py)

### 5.2 readiness / preflight

正式提交 SFT 前必须先过 readiness。

常用入口：

```bash
python scripts/cosmos3_piper14_readiness.py --require-slurm-account
```

它会检查：

- Python 解释器
- 训练 TOML
- Battery 数据
- 数据配置
- DCP 完整性
- Wan VAE
- 输出目录
- Slurm account
- 当前 W&B 模式下的凭证可见性

相关文件：

- [scripts/cosmos3_piper14_readiness.py](/project/peilab/wam/cosmos3_cy/scripts/cosmos3_piper14_readiness.py)
- [scripts/preflight_cosmos3_piper14_sft.py](/project/peilab/wam/cosmos3_cy/scripts/preflight_cosmos3_piper14_sft.py)

### 5.3 SLURM 提交训练

正式入口：

```bash
BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_cosmos3_piper14_sft.sh
```

默认行为：

- 自动先跑 readiness
- readiness 失败时不提交 `sbatch`
- 默认输出写到 `reports/cosmos3_piper14`

如果要改实验规模，优先通过环境变量或派生配置完成，不要直接把短期参数写死进主配置。

### 5.4 训练后评估

训练结束后，用离线评估脚本导出预测与对照数据：

```bash
python scripts/cosmos3_piper14_offline_eval.py \
  --checkpoint <checkpoint> \
  --run-config <config.yaml>
```

它会导出：

- `pred_action`
- `current_qpos`
- `ground_truth_action`

用于后续安全检查、误差分析和验收。

相关文件：

- [scripts/cosmos3_piper14_offline_eval.py](/project/peilab/wam/cosmos3_cy/scripts/cosmos3_piper14_offline_eval.py)
- [configs/safety/battery_piper14_safety.yaml](/project/peilab/wam/cosmos3_cy/configs/safety/battery_piper14_safety.yaml)

## 6. 当前正式 adapter 是什么

当前正式训练链路并不存在一个单独的“正式 adapter 文件”。

真正承担接入职责的是下面这一组组件：

- `piper_cosmos/cosmos3/action_policy_piper14_nano.py`
  负责注册 Cosmos3 experiment，并配置训练所需的数据、optimizer、scheduler 和 checkpoint 行为。
- `piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py`
  负责把 Piper14 HDF5 数据接成 Cosmos3 训练可用的数据集。
- `piper_cosmos/cosmos3/domain.py`
  负责注册 `piper14` domain、`domain_id` 和 action 维度约束。
- `piper_cosmos/cosmos3/train_action_policy_piper14.py`
  负责在进入官方训练模块前完成本地 HF 资产 bootstrap 和 experiment 注册。

也就是说，当前正式接入层是“Cosmos experiment 注册 + dataset + domain +  bootstrap”的组合，而不是单个 `nn.Module adapter` 文件。

## 7. 旧 placeholder adapter 已移除

旧的 `piper_cosmos/models/cosmos3_piper14_adapter.py` 是 M5 阶段的 dry-run/placeholder adapter，不在当前 Cosmos3-Nano Piper14 后训练主线上。当前正式接入层只保留 `piper_cosmos/cosmos3/` 下的 experiment 注册、domain 注册、HDF5 action dataset 和训练 bootstrap。

## 8. 对 `pack_3_objects_plus` 的处理原则

`pack_3_objects_plus` 仍然可能是未来要训练的数据，因此不能因为它不是当前 `battery_assemble` 主线就直接判定相关代码无用。

判断标准应当是：

- 代码是否仍提供通用能力
- 数据路径是否容易替换成别的数据集
- 逻辑是否仍对未来 Piper/Cosmos 训练有价值

例如：

- 通用 HDF5 reader、split 逻辑通常仍有保留价值
- 纯粹绑定旧任务、旧文件名、且没有被当前链路调用的脚本，才更接近“可清理”

## 9. 本次完成记录

本次已经完成并确认的关键事项：

- `Piper14` domain 已注册到当前 Cosmos3 实验链路
- 本地 HDF5 数据已接入训练数据集入口
- 本地 HF 资产逻辑已收敛到共享 helper
- DCP 转换链路和训练链路已统一走本地资产 bootstrap
- readiness / preflight 已覆盖 DCP、数据、输出目录、Slurm 账户和 W&B 可见性检查
- 正式训练入口已固定为 `submit_cosmos3_piper14_sft.sh`

需要持续记住的经验：

- 运行时不要依赖外网下载 tokenizer / VAE
- 不要在 batch 脚本里覆盖原始 `HOME`
- 2 卡 smoke 只能说明链路是否通，不代表正式训练规模
- Slurm 资源和 `NPROC_PER_NODE` 必须同步

## 10. 相关文档

- [docs/repo_guide.md](/project/peilab/wam/cosmos3_cy/docs/repo_guide.md)
- [docs/ACTION_SCHEMA.md](/project/peilab/wam/cosmos3_cy/docs/ACTION_SCHEMA.md)
- [docs/DATA_SCHEMA.md](/project/peilab/wam/cosmos3_cy/docs/DATA_SCHEMA.md)
- [reports/cosmos3_piper14/2026-06-30_training_pitfalls_brief_zh.md](/project/peilab/wam/cosmos3_cy/reports/cosmos3_piper14/2026-06-30_training_pitfalls_brief_zh.md)
