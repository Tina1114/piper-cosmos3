# Cosmos3 Piper14 策略部署总结

## 范围

本文总结 Cosmos3 Piper14 电池装配任务 checkpoint 的 policy server 和无真机 dry-run RTC 部署链路，覆盖：

- `/project/peilab/wam/cosmos3_cy/cosmos_battery/18k`
- `/project/peilab/wam/cosmos3_cy/cosmos_battery/20k`

前期用于 bring-up 的临时 smoke 脚本和原始 JSON report 已清理；本文保留有判断价值的测试结论。

## 核心文件

- `piper_cosmos/deployment/cosmos_piper14_policy.py`
  - 本地 Cosmos policy 封装。
  - 将 Piper14 观测转换为官方 Cosmos action policy 所需的 batch 格式。
  - 复用训练时已有的 Piper14 adapter/domain 假设。
  - 关闭 Cosmos guardrails，避免推理时访问 gated 的 `Cosmos-Guardrail1` 仓库。

- `piper_cosmos/deployment/cosmos_piper14_policy_server.py`
  - 常驻 RPC policy server。
  - 支持 `metadata`、`infer`、`update_observation`、`get_action`、`reset`、`shutdown`。
  - action 在 wire protocol 上用 Python list 传输，避免不同 Python/NumPy 环境之间 pickle 不兼容。

- `piper_cosmos/deployment/cosmos_piper14_remote_client.py`
  - 机器人侧/runtime 侧轻量 RPC client。
  - 将 server 返回的 action list 转回连续的 `np.float32` 数组。

- `piper_cosmos/deployment/piper14_rtc_runtime.py`
  - 无真机 RTC runtime 基础组件。
  - 提供 `RealTimeChunkingBuffer`、`Piper14RTCRuntime`、`HDF5ObservationSource`、`RecordingActionSink`。
  - 后续真机接入时，主要替换 `ObservationSource` 和 `ActionSink`。

- `scripts/serve_cosmos_piper14_policy.py`
  - 从导出的 checkpoint 启动 policy server。

- `scripts/dry_run_piper14_rtc_runtime.py`
  - 连接 live policy server，运行无真机 RTC loop。
  - 从 HDF5 读取真实图像帧和 qpos，请求 action chunk，每个 control step 选择一个 14 维 action，并可写出 summary report。

## 验证结论

### Policy Server

18k 和 20k 导出的 checkpoint 都可以在现有 Cosmos3 环境中加载并启动 server。server 暴露的部署 metadata 为：

- domain：`piper14`
- action type：`absolute_joint_position_command`
- action chunk shape：`[32, 14]`
- image keys：`cam_high`、`cam_left_wrist`、`cam_right_wrist`
- image size：`480x640 RGB uint8`

### HDF5 推理链路

使用真实 HDF5 数据：

`/project/peilab/wam/physical_WM/data/battery_assemble/perfect`

18k checkpoint 在采样 episode/timestep 上返回了有效 action：

- samples：`9`
- failed samples：`0`
- 合并 action shape：`[288, 14]`
- finite：`true`
- global min/max：约 `-1.81 / 2.39`
- global mean/std：约 `0.195 / 0.873`

这验证了离线真实帧推理链路：

`HDF5 RGB frames + qpos -> policy server -> [32, 14] action chunk`

### Dry-Run RTC 链路

使用 live 18k policy server 跑过无真机 RTC runtime。

在 `steps=64`、`control_hz=30`、`replan_interval=8` 下：

- selected actions：`[64, 14]`
- finite：`true`
- starved steps：`0`
- policy inferences：`8`
- 单次 action chunk 推理延迟：约 `3.47s`

这验证了软件部署链路：

`HDF5ObservationSource -> remote policy client -> RTC chunk buffer -> selected [14] action -> RecordingActionSink`

## 部署判断

policy server 和无真机 dry-run RTC 软件链路已经跑通。当前实现能够加载导出的 Cosmos checkpoint，消费真实 Piper14 HDF5 观测，并输出有限的 14 维绝对关节位置命令 chunk。

但当前 Cosmos 推理速度不满足 30 Hz 高频闭环控制预算。30 Hz 下：

- 8 步 replan 窗口约 `0.27s`
- 32 步 action chunk 约 `1.07s`
- 实测单次 chunk 推理约 `3.47s`

因此，当前版本不能直接当作 30 Hz 高频闭环真机控制器使用。真机部署需要采用更保守的策略，例如低频 chunk 执行、异步预取、推理加速，或者先做 supervised/no-motion 验证。

下一步有价值的工作不是继续做离线 smoke 参数组合，而是真机 adapter 边界：

- `ObservationSource`：真实同步相机帧和 Piper14 qpos。
- `ActionSink`：带保护的绝对关节位置命令发送器。
- 安全检查：action shape、有限值、关节限位、单步 action jump、state tracking error。

初次真机验证建议先只读取真实观测并计算 action chunk，不使能运动；确认观测、推理、安全检查稳定后，再进入受保护的低频/chunk 执行。

## 使用命令

在 GPU 节点启动 18k policy server：

```bash
cd /project/peilab/wam/cosmos3_cy

PYTHONPATH=/project/peilab/wam/cosmos3_cy:/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3 \
external/cosmos/packages/cosmos3/.venv/bin/python \
scripts/serve_cosmos_piper14_policy.py \
  --checkpoint /project/peilab/wam/cosmos3_cy/cosmos_battery/18k \
  --host 0.0.0.0 \
  --port 8766
```

从登录节点或另一个终端运行无真机 dry-run RTC：

```bash
cd /project/peilab/wam/cosmos3_cy

PYTHONPATH=/project/peilab/wam/cosmos3_cy \
python scripts/dry_run_piper14_rtc_runtime.py \
  --host dgx-51 \
  --port 8766 \
  --data-root /project/peilab/wam/physical_WM/data/battery_assemble/perfect \
  --data-config configs/dataset_configs/battery_assemble_hdf5.yaml \
  --steps 64 \
  --control-hz 30 \
  --replan-interval 8 \
  --report reports/cosmos3_piper14/rtc_dry_run_latest.json
```
