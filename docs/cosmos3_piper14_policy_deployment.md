# Cosmos3 Piper14 Policy Server 部署说明

## 目标

本分支实现 Cosmos3 Piper14 battery_assemble checkpoint 的推理部署基础链路：

- 启动常驻 policy server。
- 通过 RPC 从远端请求 action chunk。
- 支持无真机 dry-run RTC runtime，用 HDF5 真实帧模拟机器人观测。

当前覆盖 checkpoint：

- `/project/peilab/wam/cosmos3_cy/cosmos_battery/18k`
- `/project/peilab/wam/cosmos3_cy/cosmos_battery/20k`

## 核心文件

- `piper_cosmos/deployment/cosmos_piper14_policy.py`
  - 本地 policy 封装。
  - 把 Piper14 观测转换成 Cosmos3 action policy 推理 batch。
  - 关闭 Cosmos guardrails，避免推理时访问 gated Guardrail1 仓库。

- `piper_cosmos/deployment/cosmos_piper14_policy_server.py`
  - 常驻 RPC server。
  - 支持 `metadata`、`infer`、`update_observation`、`get_action`、`reset`、`shutdown`。

- `piper_cosmos/deployment/cosmos_piper14_remote_client.py`
  - 远端 RPC client。
  - 将 server 返回的 action list 转成 `np.float32` 数组。

- `piper_cosmos/deployment/piper14_rtc_runtime.py`
  - 无真机 RTC runtime。
  - 包含 action chunk buffer、HDF5 observation source、recording action sink。
  - 真机接入时主要替换 `ObservationSource` 和 `ActionSink`。

- `scripts/serve_cosmos_piper14_policy.py`
  - 启动 policy server。

- `scripts/dry_run_piper14_rtc_runtime.py`
  - 连接 live server，用 HDF5 数据跑无真机 RTC loop。

## 已验证结论

- 18k 和 20k checkpoint 均可加载并启动 server。
- server metadata：
  - domain：`piper14`
  - action type：`absolute_joint_position_command`
  - action chunk：`[32, 14]`
  - image keys：`cam_high`、`cam_left_wrist`、`cam_right_wrist`
  - image size：`480x640 RGB uint8`
- HDF5 真实帧推理通过：
  - 9 个采样点全部成功。
  - 输出 action 合并 shape 为 `[288, 14]`。
  - action 全部 finite，无 NaN/Inf。
- dry-run RTC 通过：
  - `steps=64`
  - `replan_interval=8`
  - `selected_actions=[64, 14]`
  - `starved_steps=0`
  - 单次 Cosmos action chunk 推理约 `3.47s`

## 重要判断

部署软件链路已经跑通：

```text
HDF5/机器人观测 -> RPC client -> policy server -> [32,14] action chunk -> RTC action buffer -> [14] action
```

但当前 Cosmos 推理速度不满足 30 Hz 高频闭环真机控制。30 Hz 下：

- 8 步窗口约 `0.27s`
- 32 步 chunk 约 `1.07s`
- 实测一次推理约 `3.47s`

因此当前版本适合做：

- policy server 部署验证
- HDF5 无真机 RTC 验证
- 真机 no-motion/supervised 预检查

不应直接当作 30 Hz 高频闭环控制器使用。

## 使用命令

在 GPU 节点启动 18k server：

```bash
cd /project/peilab/wam/cosmos3_cy

PYTHONPATH=/project/peilab/wam/cosmos3_cy:/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3 \
external/cosmos/packages/cosmos3/.venv/bin/python \
scripts/serve_cosmos_piper14_policy.py \
  --checkpoint /project/peilab/wam/cosmos3_cy/cosmos_battery/18k \
  --host 0.0.0.0 \
  --port 8766
```

从登录节点或另一个终端跑 dry-run RTC：

```bash
cd /project/peilab/wam/cosmos3_cy

PYTHONPATH=/project/peilab/wam/cosmos3_cy \
python scripts/dry_run_piper14_rtc_runtime.py \
  --host <gpu-node-hostname> \
  --port 8766 \
  --data-root /project/peilab/wam/physical_WM/data/battery_assemble/perfect \
  --data-config configs/dataset_configs/battery_assemble_hdf5.yaml \
  --steps 64 \
  --control-hz 30 \
  --replan-interval 8 \
  --report reports/cosmos3_piper14/rtc_dry_run_latest.json
```

## 下一步

真机接入不需要继续扩展 smoke 测试。下一步应实现：

- 真实 `ObservationSource`：同步相机帧 + Piper14 qpos。
- 真实 `ActionSink`：带安全检查的绝对关节位置命令发送。
- 安全保护：shape、finite、关节限位、单步 action jump、state tracking error。
