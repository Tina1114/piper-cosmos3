# Cosmos3 Battery Assemble 真机数据 SFT 计划

本文规划如何在不破坏 FastWAM 环境和原始数据的前提下，把 Battery Assemble 真机数据分别接入 `Cosmos3-Nano` 与 `Cosmos3-Nano-Policy-DROID` 做 action-policy SFT。

## 结论

- 主路线使用 `Cosmos3-Nano -> Piper14 SFT`。
- `Cosmos3-Nano-Policy-DROID` 只作为 warm-start 对比路线；不能直接使用 DROID 8D action head 或 `droid_lerobot` domain。
- Battery action 必须按 [ACTION_SCHEMA.md](/project/peilab/wam/cosmos3_cy/docs/ACTION_SCHEMA.md) 解释为 Piper 双臂 `14D` 绝对关节目标，不能解释为 delta、EEF，也不能填充为 Cosmos 官方 dual-DROID `20D`。
- 数据、checkpoint、cache、输出都放在 `external/cosmos/` 或显式 scratch 路径；不修改 FastWAM 环境和 `/project/peilab/wam/physical_WM/data/battery_assemble` 原始数据。

## 官方依据

- `external/cosmos/README.md` 说明 Cosmos3 Generator 支持 text、vision、action，并覆盖 action policy、inverse dynamics、forward dynamics。
- `external/cosmos/packages/cosmos3/docs/action_policy_droid_posttrain.md` 说明 `Cosmos3-Nano-Policy-DROID` 是从 `Cosmos3-Nano` 在 `Cosmos3-DROID` 上 SFT 得到。
- `external/cosmos/packages/cosmos3/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py` 给出了官方 DROID recipe：LeRobotDataset v3.0、DROID `joint_pos` 8D、`use_state=true`、`concat_view`、chunk length 32、count-based batch、FSDP/HSDP、多 GPU 训练。

## Battery 数据事实

Raw 数据：

- 路径：`/project/peilab/wam/physical_WM/data/battery_assemble/perfect`
- HDF5 字段：`action [T,14]`、`qpos [T,14]`、三路 RGB `480x640`
- `perfect/` split 已确认 `action[t]` 最接近 `qpos[t+1]`，因此第一阶段 policy 输出是下一时刻绝对关节目标。

LeRobot 导出：

- 路径：`/project/peilab/wam/physical_WM/data/battery_assemble/perfect_lerobot`
- `meta/info.json`：
  - `codebase_version=v2.1`
  - `total_episodes=138`
  - `total_frames=125611`
  - `fps=30`
  - `action [14]`
  - `observation.state [14]`
  - 三路视频：`observation.images.cam_high`、`observation.images.cam_left_wrist`、`observation.images.cam_right_wrist`
- `meta/tasks.jsonl`：
  - task：`Assemble the mouse's battery.`

动作维度遵循 `docs/ACTION_SCHEMA.md`：

| 维度 | 名称 |
| --: | ---- |
| 0 | left_waist |
| 1 | left_shoulder |
| 2 | left_elbow |
| 3 | left_forearm_roll |
| 4 | left_wrist_angle |
| 5 | left_wrist_rotate |
| 6 | left_gripper |
| 7 | right_waist |
| 8 | right_shoulder |
| 9 | right_elbow |
| 10 | right_forearm_roll |
| 11 | right_wrist_angle |
| 12 | right_wrist_rotate |
| 13 | right_gripper |

## 环境策略

FastWAM 在本计划中的角色只有两个：

- 作为 Battery 原始数据和历史机器人代码所在项目的边界名称。
- 必要时作为只读参考环境临时跑本地 CPU 数据检查。

它不是 Cosmos3 SFT 的默认训练环境。主验收脚本默认使用 `external/cosmos/packages/cosmos3/.venv/bin/python`；如 GPU 节点需要不同解释器，必须通过 `PYTHON_BIN=/path/to/python` 显式指定。不要把 Cosmos3 依赖安装进 FastWAM，也不要用训练脚本修改 FastWAM 环境。

现有检查结果：

- `external/cosmos/packages/cosmos3/.venv` 已存在。
- `cosmos_framework` 可 import。
- 当前登录环境曾观测到 `torch=2.10.0+cu128`、`torch.version.cuda=12.8`。
- 登录 shell 中 `torch.cuda.is_available() == False` 只说明当前 shell 没有 GPU，不直接判定训练环境不可用。

执行原则：

- FastWAM 环境只读；不改、不安装包、不激活写入。
- 优先复用 `external/cosmos/packages/cosmos3/.venv`。
- 如果 GPU 节点上 `cosmos_framework`、`torchrun`、`flash-attn` 或 `transformer-engine` 失败，只在 Cosmos 目录处理环境：
  - 优先在 `external/cosmos/packages/cosmos3` 内按官方方式执行 `uv sync --all-extras --group=cu128-train` 重建同一路径 `.venv`。
  - 或创建 `external/cosmos/envs/cosmos3-cu128-train` 作为可复用新环境。
- `HF_HOME`、模型 cache、DCP checkpoint、训练输出、日志和导出权重都放在 `external/cosmos/` 或显式 scratch 路径。

GPU 探测顺序：

1. 申请 1 GPU：运行 `nvidia-smi`、torch CUDA import、`cosmos_framework` import。
2. 申请 2 GPU：跑 dataset + model forward/backward smoke。
3. 申请 4 GPU：跑 10-50 iter SFT smoke，确认显存、checkpoint 保存和 resume 基本路径。

## 路线 A：Cosmos3-Nano -> Piper14 SFT

这是默认主路线。

### 数据接入

- 优先读取 `perfect_lerobot`，因为它已经具备 parquet、video、meta、fps 和 task 信息。
- 新增 `Piper14LeRobotDataset` 或同等 dataset factory，读取：
  - `action`
  - `observation.state`
  - `observation.images.cam_high`
  - `observation.images.cam_left_wrist`
  - `observation.images.cam_right_wrist`
  - instruction/task：`Assemble the mouse's battery.`
- 如果官方 LeRobot v3 reader 不能直接读取当前 `codebase_version=v2.1` layout，则按 `meta/info.json` 的 `data_path` 和 `video_path` 自己读取 parquet/video；不得修改原始数据。

### Cosmos action 接口

- 新增 `domain_name="piper14"`。
- 注册 `raw_action_dim=14` 和独立 `domain_id`。
- 复用官方 action padding/masking 机制，但训练、评估、导出和部署只信任前 14 维。
- 不使用 DROID `joint_pos` 8D action space，也不套 Cosmos 官方 dual-arm DROID 20D schema。

### Split 和窗口

- 第一版使用 `perfect_lerobot` 全量成功数据。
- 按 episode 做 `90% train / 10% val`，避免同一 episode 的重叠窗口同时进入 train 和 val。
- 不使用 DROID `keep_ranges_1_0_1.json`。
- 后续可以基于 Piper14 action/qpos 活动量生成 Battery 自己的 active-window filter，但不能直接套 DROID filter。

### SFT 配置

- 初始化：从 `nvidia/Cosmos3-Nano` 转换后的 DCP checkpoint 加载。
- chunk length：默认 32，对应官方 tokenizer `encode_exact_durations=[33]`。
- fps：先使用 30；如果显存或吞吐不稳，可以创建只读下采样 dataset view 到 15 fps，但必须同步记录 action-time 对齐。
- state：`use_state=true`，使用初始 `qpos [14]` 作为 proprio/state 条件。
- action normalization：
  - 第一版使用 raw absolute joint position，不套 DROID 归一化。
  - 如果 action loss、输出范围或 gripper 维度不稳，再按 Battery train split 统计 mean/std，并保存 inverse transform 与 checkpoint 绑定。
- optimizer/加载策略参考 DROID recipe：
  - 加载 backbone、视觉/生成相关权重。
  - 跳过 action head/action embedding/action projection 相关权重，让 Piper14 action head fresh init。
  - 需要显式记录 skipped keys，例如 `action2llm`、`llm2action`、`action_modality_embed`、`action_pos_embed` 及新增 Piper14 domain 相关参数。

### Batch 和显存计划

官方 DROID 参考是大规模 global batch 8192。2/4 GPU 不能真实复现该规模，只复现 count-based batch + grad accumulation 的训练形式。

- 2 GPU smoke：
  - `max_samples_per_batch=4~8`
  - `grad_accum_iter=1`
  - chunk 32
  - 目标只验证 dataset、前向、反向、loss 非 NaN。
- 4 GPU smoke：
  - `max_samples_per_batch=8~16`
  - `grad_accum_iter=2`
  - 有效 batch 约 `4 * max_samples_per_batch * 2 = 64~128`
  - 跑 10-50 iter 并保存 checkpoint。
- 如果 4 GPU H800 80GB 稳定：
  - 提升到 `max_samples_per_batch=32`
  - 有效 batch 约 256。
- 正式训练优先 4/8 GPU；16B bf16 + 33 帧 480p activation 显存较重，2 GPU 只作为功能验证，不作为正式 SFT 配置。

### 验收标准

- 10 iter smoke：
  - loss 有效、无 NaN。
  - 可以保存 checkpoint。
  - 导出的 sample output 只有 Piper14 前 14 维。
- 500 iter pilot：
  - val action MAE/MSE 下降。
  - 输出范围接近 train split 数据统计。
  - gripper 维度 6/13 的误差和越界率单独报告。
- 正式 run：
  - 使用全量 train split。
  - 保存 DCP checkpoint。
  - 导出可推理权重。

## 路线 B：Policy-DROID warm-start -> reset action heads -> Piper14 SFT

这条路线只作为对比实验，不作为默认主线。

### 为什么不能直接使用 Policy-DROID

- `Cosmos3-Nano-Policy-DROID` 绑定 DROID `joint_pos` 8D action 和 DROID camera/viewpoint 分布。
- Battery 是 Piper 双臂 `14D`；不能直接使用 `droid_lerobot` domain。
- 不能把 Battery 14D 截断、重排或映射成 DROID 8D。
- 不能把 Battery 14D padding 成 DROID dual-arm 20D 后按 DROID 语义训练。

### 正确接入方式

- 加载 `Cosmos3-Nano-Policy-DROID` checkpoint 中可迁移的通用视觉/生成/backbone 权重。
- 跳过或重置所有 DROID action 维度绑定权重：
  - action projection
  - action head
  - action modality embedding
  - domain/action stats
  - DROID domain id/action-space 绑定参数
- 使用和路线 A 完全相同的 `piper14` dataset、domain、split、chunk、batch、eval 和 safety filter。

### 对比实验

- A：`Cosmos3-Nano -> Piper14 SFT`
- B：`Cosmos3-Nano-Policy-DROID -> reset action heads -> Piper14 SFT`
- 两者必须保持：
  - 同一 train/val episode split
  - 同一 chunk length
  - 同一 batch/grad accumulation
  - 同一随机种子
  - 同一 safety/offline eval
- 如果 B 在 500-1000 iter 后 val loss/MAE 明显差于 A，停止 B，主线保留 A。

## Safety Filter 计划

不能完全信任现有 safety filter。真机前必须基于 Battery train split 重算约束。

统计项：

- 每个 action 维度的 min/max、p01/p99、mean/std。
- 相邻 action delta 分布。
- gripper 维度 6/13 的合法范围和轻微负值情况。
- `docs/ACTION_SCHEMA.md` 已记录 gripper 部署裁剪到 `[0.0, 0.1]`，且 `perfect/` 中存在轻微负值。

默认策略：

- hard limit 使用机器人真实关节/夹爪部署约束。
- soft limit 使用 Battery train split p01/p99 加 margin。
- delta limit 使用数据统计分位数，先保守，再根据离线回放调整。
- safety 失败时 hold current qpos，不发送异常目标。

验收：

- 对 val 输出批量跑 safety report。
- 报告越界维度、越界幅度、clip 次数、hold 次数。
- 真机前先 shadow mode，只记录预测和 safety 决策。
- 通过离线安全报告后，才允许显式 execute。

## 测试计划

### 训练验收门槛

训练完成后必须用 `scripts/validate_training_acceptance.py` 生成 acceptance report。该 report 是是否通过训练验收的机器可读依据，不能只看最后一步 loss。

最小输入：

```shell
python scripts/validate_training_acceptance.py \
  --training reports/battery_piper14/training_report.json \
  --eval reports/battery_piper14/eval.json \
  --safety reports/battery_piper14/safety_report.json \
  --predictions reports/battery_piper14/validation_predictions.npz \
  --report reports/battery_piper14/acceptance_report.json
```

`acceptance_report.json` 必须全部通过以下 gate：

- `train_loss`：train loss 有限，滑动平均下降，无 NaN/Inf。
- `validation_loss`：validation loss 相比初始值明显下降，且 overfit gap 不超过阈值。
- `small_batch_overfit`：极小数据 overfit 测试达到低误差阈值。
- `per_dim_action_error`：14 个 Piper action 维度分别报告 MAE/MSE。
- `safety_limits`：validation 预测基本不违反 joint limit、gripper limit 和 max delta。
- `trajectory_trend`：预测轨迹和 ground truth 趋势相关，不是均值预测或零动作预测。
- `checkpoint_artifacts`：checkpoint 可 resume/inference，且保存 config、normalization stats、dataset split。
- `best_checkpoint`：至少一个 best checkpoint 由 validation metric 选出，而不是按最后一步 train loss 选出。
- `run_scope`：正式验收必须是 `battery_piper14_full`，至少完成 `500` 个 optimizer step，使用 `224` 图像尺寸，不启用 `limit_samples`，且 train/val 样本、Piper14 `14D` action 和 action horizon 记录完整。CPU/64px/2-step smoke 即使其它 gate 通过，也不能作为正式验收通过。
- `model_backend`：正式验收必须声明真实 `cosmos3` backend，base model 是 `nvidia/Cosmos3-Nano` 或 `nvidia/Cosmos3-Nano-Policy-DROID`，`real_cosmos3_backbone=true`，`placeholder_backbone=false`，action head 为 `piper14` 且 `action_dim=14`。`SimpleMultiViewCNNPolicy` baseline 和 `PlaceholderCosmos3Backbone` adapter 只能用于管线验证，不能满足最终 SFT 验收。

完整验收流水线由 `scripts/run_battery_piper14_acceptance_slurm.sh` 执行：

```shell
BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_battery_piper14_acceptance.sh
```

真实 Cosmos3 SFT 入口为 `scripts/run_cosmos3_piper14_sft_slurm.sh`，它使用本仓库注册的 `action_policy_piper14_nano` experiment，而不是 baseline trainer：

```shell
export PIPER14_ROOT=/project/peilab/wam/physical_WM/data/battery_assemble/perfect
export PIPER14_DATA_CONFIG=/project/peilab/wam/cosmos3_cy/configs/dataset_configs/battery_assemble_hdf5.yaml
export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP checkpoint dir>
export WAN_VAE_PATH=<Wan2.2_VAE.pth>
export BATTERY_SLURM_ACCOUNT=<account>
bash scripts/submit_cosmos3_piper14_sft.sh
```

`scripts/submit_cosmos3_piper14_sft.sh` 会先运行 `scripts/cosmos3_piper14_readiness.py`。readiness 汇总 DCP verifier 和 SFT preflight；失败时不会提交 `sbatch`。如果需要只做底层 SFT preflight 诊断：

```shell
python scripts/preflight_cosmos3_piper14_sft.py \
  --base-checkpoint "${BASE_CHECKPOINT_PATH}" \
  --wan-vae "${WAN_VAE_PATH}" \
  --require-slurm-account \
  --report reports/cosmos3_piper14/preflight.json
```

当 `BASE_CHECKPOINT_PATH` 或 `WAN_VAE_PATH` 未设置时，preflight 会在失败报告里尽量给出本地候选路径。当前工作区已经发现 `external/cosmos/checkpoints/Cosmos3-Nano` HF checkpoint 和 `external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/.../Wan2.2_VAE.pth`。登录节点尝试 DCP 转换时已经确认会卡在无 NVIDIA driver 的 VAE 初始化，因此 DCP 转换应提交到 GPU 节点：

```shell
export BATTERY_SLURM_ACCOUNT=<account>
bash scripts/submit_convert_cosmos3_nano_dcp.sh
```

该提交脚本调用 `scripts/run_convert_cosmos3_nano_dcp_slurm.sh`，后者使用 `scripts/convert_cosmos3_nano_to_dcp_offline.py` 复用本地 Qwen snapshot，避免转换过程中因 `uvx hf download` 或 PyPI 网络受限失败。转换完成后，设置：

```shell
export BASE_CHECKPOINT_PATH=/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
```

`scripts/preflight_cosmos3_piper14_sft.py` 不只检查 DCP 目录存在，还会要求 `BASE_CHECKPOINT_PATH/model/.metadata` 和至少一个 `*.distcp` shard 存在，避免把失败后的半成品目录当作可训练 checkpoint。

转换作业末尾会运行：

```shell
python scripts/verify_cosmos3_dcp.py \
  --path external/cosmos/checkpoints/Cosmos3-Nano-DCP \
  --report reports/cosmos3_piper14/dcp_verify.json
```

只有 `reports/cosmos3_piper14/dcp_verify.json` 的 `status=passed` 时，才继续提交 Piper14 SFT。

当前门槛状态可以用一个统一摘要脚本查看；它不会提交作业，只汇总 DCP verifier、最近一次 readiness JSON 和转换 job 状态：

```shell
python scripts/cosmos3_piper14_status.py \
  --dcp-path external/cosmos/checkpoints/Cosmos3-Nano-DCP \
  --readiness-report reports/cosmos3_piper14/readiness.json \
  --conversion-job-id <job_id> \
  --conversion-job-state <PENDING|RUNNING|COMPLETED|FAILED> \
  --conversion-job-reason <reason> \
  --conversion-start-time <start_time_or_unknown> \
  --report reports/cosmos3_piper14/status.json
```

如果输出 `next_action=wait_for_dcp_conversion`，说明 DCP 仍未完整，不能提交 SFT；如果输出 `next_action=run_readiness`，先重新跑 readiness；只有 `next_action=submit_sft` 且 `ready_for_sft=true` 时才提交训练。

提交 SFT 前可以用 readiness 汇总脚本同时检查 DCP、VAE、数据、TOML、Cosmos3 Python、输出目录和 SLURM account：

```shell
python scripts/cosmos3_piper14_readiness.py \
  --base-checkpoint "${BASE_CHECKPOINT_PATH}" \
  --wan-vae "${WAN_VAE_PATH}" \
  --require-slurm-account \
  --report reports/cosmos3_piper14/readiness.json
```

只有 `reports/cosmos3_piper14/readiness.json` 的 `ready_for_sft=true` 时，才执行 `bash scripts/submit_cosmos3_piper14_sft.sh`。

真实 Cosmos3 训练完成后，先把 Cosmos3 指标和 checkpoint 汇总成 acceptance gate 使用的 training report。示例：

```shell
python scripts/cosmos3_piper14_training_report.py \
  --metrics reports/cosmos3_piper14/<run>/metrics.jsonl \
  --checkpoint reports/cosmos3_piper14/<run>/checkpoints/<latest> \
  --best-checkpoint reports/cosmos3_piper14/<run>/checkpoints/<best-val> \
  --config reports/cosmos3_piper14/<run>/config.yaml \
  --normalization-stats reports/battery_assemble_dataset_stats_perfect.json \
  --dataset-split reports/battery_assemble_dataset_split.json \
  --base-model nvidia/Cosmos3-Nano \
  --route nano \
  --image-size 480 \
  --action-horizon 32 \
  --train-samples 124 \
  --val-samples 14 \
  --optimizer-steps 500 \
  --output reports/battery_piper14/training_report.json
```

该脚本只生成训练侧报告字段：loss 序列、best validation checkpoint、run scope、Cosmos3 backend 声明和 artifact 路径。`resume_check`、`inference_check`、`small_batch_overfit` 如果没有显式提供 JSON artifact，会保守标记为 `not_run`，因此不会让最终 acceptance gate 误通过。后续仍必须对真实 Cosmos3 checkpoint 跑 offline eval、prediction export、safety report 和 `scripts/validate_training_acceptance.py`。

如果真实 Cosmos3 checkpoint 的 eval、safety 和 prediction artifacts 已经生成，可以用一条命令把 training report 和最终 acceptance report 一起产出：

```shell
python scripts/cosmos3_piper14_acceptance_pipeline.py \
  --metrics reports/cosmos3_piper14/<run>/metrics.jsonl \
  --checkpoint reports/cosmos3_piper14/<run>/checkpoints/<latest> \
  --best-checkpoint reports/cosmos3_piper14/<run>/checkpoints/<best-val> \
  --config reports/cosmos3_piper14/<run>/config.yaml \
  --normalization-stats reports/battery_assemble_dataset_stats_perfect.json \
  --dataset-split reports/battery_assemble_dataset_split.json \
  --eval reports/battery_piper14/eval.json \
  --safety reports/battery_piper14/safety_report.json \
  --predictions reports/battery_piper14/validation_predictions.npz \
  --base-model nvidia/Cosmos3-Nano \
  --route nano \
  --image-size 480 \
  --action-horizon 32 \
  --train-samples 124 \
  --val-samples 14 \
  --optimizer-steps 500 \
  --output-dir reports/battery_piper14
```

该 orchestrator 只编排已有真实产物：缺少 metrics、checkpoint、eval、safety 或 predictions 时会直接失败；不会用 baseline report 或占位输出替代真实 Cosmos3 验收。

如果真实 Cosmos3 checkpoint 的 offline eval、prediction export 和 safety report 已经生成，可以用 post-training pipeline 一次性写出 `training_report.json` 和 `acceptance_report.json`：

```shell
python scripts/cosmos3_piper14_acceptance_pipeline.py \
  --metrics reports/cosmos3_piper14/<run>/metrics.jsonl \
  --checkpoint reports/cosmos3_piper14/<run>/checkpoints/<latest> \
  --best-checkpoint reports/cosmos3_piper14/<run>/checkpoints/<best-val> \
  --config reports/cosmos3_piper14/<run>/config.yaml \
  --normalization-stats reports/battery_assemble_dataset_stats_perfect.json \
  --dataset-split reports/battery_assemble_dataset_split.json \
  --eval reports/battery_piper14/eval.json \
  --safety reports/battery_piper14/safety_report.json \
  --predictions reports/battery_piper14/validation_predictions.npz \
  --base-model nvidia/Cosmos3-Nano \
  --route nano \
  --image-size 480 \
  --action-horizon 32 \
  --train-samples 124 \
  --val-samples 14 \
  --optimizer-steps 500 \
  --output-dir reports/battery_piper14
```

该 pipeline 只编排已有真实 artifact：它不会替代 Cosmos3 inference，不会生成假的 eval/safety/prediction，也不会跳过 gate。缺少任一输入文件会直接失败；`acceptance_report.json` 任一 gate 失败时命令退出非零。

对应 TOML 是 `configs/cosmos3/sft/action_policy_piper14_nano.toml`。该 experiment：

- 注册 `domain_name=piper14`，`domain_id=21`，`raw_action_dim=14`。
- 从 Battery HDF5 读取三路 RGB、`qpos [14]` 和 `action [14]`。
- 用官方 `ActionTransformPipeline` 做 video resize/padding、prompt augmentation、action padding/masking。
- 使用 `chunk_length=32`，`use_state=true`，action 序列为 `[initial qpos] + 32` 个 Piper14 绝对关节目标。
- 从 `BASE_CHECKPOINT_PATH` 加载 Cosmos3-Nano DCP，并跳过 action head/action embedding 相关权重。

`scripts/run_battery_piper14_acceptance_slurm.sh` 目前仍是 baseline/pipeline 验收辅助脚本；它可以验证 split/stats/eval/safety/acceptance 机制，但不能满足最终 `model_backend` gate。最终正式验收必须由真实 Cosmos3 SFT 产物生成 `reports/battery_piper14/acceptance_report.json`。

提交前可以先跑 preflight，检查 Cosmos3 Python、配置、Battery 数据、split/stats、SLURM account 和正式验收报告状态：

```shell
python scripts/preflight_battery_piper14_acceptance.py \
  --require-slurm-account \
  --report reports/battery_piper14/preflight.json
```

`scripts/submit_battery_piper14_acceptance.sh` 会自动执行同一 preflight；preflight 不通过时不会调用 `sbatch`。如果要审计正式训练是否已经通过，再加 `--require-acceptance-report`，此时 `reports/battery_piper14/acceptance_report.json` 必须存在且所有 gate 为 passed。

默认输出目录为 `reports/battery_piper14/`，其中 `acceptance_report.json` 是正式验收依据。可以用环境变量调整试运行规模，例如：

```shell
BATTERY_SLURM_ACCOUNT=<account> \
MAX_STEPS=50 \
MAX_VAL_BATCHES=8 \
EVAL_BATCHES=8 \
bash scripts/submit_battery_piper14_acceptance.sh
```

当前本地 CPU smoke 只能证明 pipeline、acceptance gates、checkpoint/eval/export/safety 报告路径能跑通；它不能替代 GPU 节点上的正式 `reports/battery_piper14/acceptance_report.json`。

### 环境测试

- 登录节点：
  - `.venv` 可 import `cosmos_framework`。
  - 不要求 `torch.cuda.is_available()` 为 true。
- 1 GPU：
  - `nvidia-smi` 正常。
  - torch CUDA 可用。
  - `cosmos_framework` 训练入口可 import。
  - `flash-attn`、`transformer-engine` 可 import 或按官方环境重建。

### 数据测试

- 读取 `perfect_lerobot/meta/info.json`，断言：
  - `total_episodes == 138`
  - `total_frames == 125611`
  - `fps == 30`
  - `action` shape 为 `[14]`
  - `observation.state` shape 为 `[14]`
  - 三路视频存在且为 RGB video。
- 抽样 dataset batch，断言：
  - video 非空。
  - instruction 为 `Assemble the mouse's battery.`
  - action/state 均为 14D。
  - batch 中 episode split 不泄漏。

### 训练测试

- 2 GPU forward/backward smoke：
  - loss 有效。
  - backward 成功。
  - 梯度无 NaN/Inf。
- 4 GPU 10-50 iter SFT smoke：
  - `grad_accum_iter=2`。
  - checkpoint 保存成功。
  - 可以从 latest checkpoint resume。
- 500 iter pilot：
  - 路线 A 与路线 B 对比。
  - val MAE/MSE、gripper MAE、越界率、安全通过率都有记录。

### 导出和评估

- 导出 `sample_outputs.json` 时只包含 14D Piper action。
- val eval 至少包含：
  - action MAE/MSE
  - gripper MAE
  - 每维 p01/p99 范围对比
  - safety pass rate
  - clip/hold 次数

## 实施顺序

1. 冻结数据和环境边界：
   - 只读 Battery raw/LeRobot 数据。
   - 不改 FastWAM 环境。
   - 设置 Cosmos cache/checkpoint/output 路径。
2. 写 Battery metadata validator：
   - 校验 episode/frame/fps/action/state/video/task。
3. 写 Piper14 dataset/domain：
   - 支持 LeRobot v2.1 layout。
   - 输出 Cosmos action-policy 所需样本。
4. 新增 `action_policy_piper14_nano` experiment：
   - 复用 DROID recipe 结构。
   - 替换 dataset/domain/action_dim。
   - 配置 skip-load action 相关权重。
5. 先跑环境和数据 smoke。
6. 跑 2 GPU 前后向 smoke。
7. 跑 4 GPU 10-50 iter smoke。
8. 跑 500 iter pilot，对比 Nano baseline 与 Policy-DROID warm-start。
9. 生成 safety report。
10. 决定正式训练配置并导出推理权重。

## Assumptions

- 默认只使用 Battery `perfect` 成功数据，不混入 `retry` 或失败数据。
- 默认 instruction 为 `Assemble the mouse's battery.`。
- 默认优先 4 GPU 做可训练 smoke；2 GPU 只用于排错。
- 若现有 Cosmos3 `.venv` 在 GPU 节点可用，不重建环境。
- 若环境不可用，只修改或重建 Cosmos3 环境，不触碰 FastWAM。
