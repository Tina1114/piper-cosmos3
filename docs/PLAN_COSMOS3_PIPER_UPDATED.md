# Cosmos3 + Dual-Piper 真机数据复现到部署计划

## 0. 当前目标

目标是使用双臂 Piper 真机采集数据，训练或适配 Cosmos3 / Cosmos3-Piper14 policy，并最终部署到真机上进行低速安全推理。

第一阶段目标不是直接做 Cosmos3 dual-arm 20D EEF action，而是先跑通：

```text
三路 RGB 图像 + qpos + instruction
→ policy
→ 14D absolute joint-position command
→ safety filter
→ Piper 双臂真机低速执行
```

第一阶段原则：

```text
1. 不覆盖原始 HDF5。
2. 不把 RGB 当 BGR。
3. 不把 action 当 joint delta。
4. 不把 action 当 EEF pose。
5. 不把 14D action padding 成 20D。
6. 不丢掉 qpos 输入。
7. 真机脚本默认 shadow mode。
8. 真机执行必须显式传 --execute。
9. safety filter fail 时必须 hold position。
```

---

## 1. 已确认数据结构

样例文件：

```text
/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect/episode_0.hdf5
```

HDF5 attrs：

```text
compress = False
sim = False
```

HDF5 key 结构：

```text
/action                                      float32 (894, 14)
/base_action                                 float32 (894, 2)    # 全 0，忽略
/observations/effort                         float32 (894, 14)
/observations/qpos                           float32 (894, 14)
/observations/qvel                           float32 (894, 14)
/observations/images/cam_high                uint8   (894, 480, 640, 3)
/observations/images/cam_left_wrist          uint8   (894, 480, 640, 3)
/observations/images/cam_right_wrist         uint8   (894, 480, 640, 3)
```

图像已确认：

```text
layout: THWC
dtype: uint8
color: RGB
resolution: 480 x 640
cameras:
  - cam_high
  - cam_left_wrist
  - cam_right_wrist
```

动作已确认：

```text
/action.shape = [894, 14]
action type = absolute joint-position command
action[t] ≈ qpos[t+1]
mean absolute difference ≈ 0.00095 on episode_0
```

任务文本：

```text
原 HDF5 内没有 language/task/success key。

LeRobot metadata 中有 task instruction:
"Put the three objects on the table into the container."
```

成功标签：

```text
原 HDF5 内没有 success key。
路径包含 perfect/，可在 dataset level 视作 success demo。
```

---

## 2. 第一阶段技术决策

第一阶段不转 Cosmos3 dual-arm 20D。

使用原始 14D action 作为训练和部署动作接口。

训练输入：

```text
cam_high[t]
cam_left_wrist[t]
cam_right_wrist[t]
qpos[t]
optional qvel[t]
instruction
```

训练目标：

```text
action[t:t+H]  # [H, 14], absolute joint-position command
```

部署输出：

```text
predicted_action[0]  # [14], next absolute joint-position target
```

部署执行：

```text
current_qpos
predicted absolute joint target
→ safety filter
→ safe joint target
→ send joint-position command
```

20D EEF action conversion 作为第二阶段研究任务，不作为当前 MVP 阻塞项。

---

## 3. 14D Action Schema

动作顺序：

| dim | name               |
| --: | ------------------ |
|   0 | left_waist         |
|   1 | left_shoulder      |
|   2 | left_elbow         |
|   3 | left_forearm_roll  |
|   4 | left_wrist_angle   |
|   5 | left_wrist_rotate  |
|   6 | left_gripper       |
|   7 | right_waist        |
|   8 | right_shoulder     |
|   9 | right_elbow        |
|  10 | right_forearm_roll |
|  11 | right_wrist_angle  |
|  12 | right_wrist_rotate |
|  13 | right_gripper      |

episode_0 action range：

| dim | name               |       min |       max |
| --: | ------------------ | --------: | --------: |
|   0 | left_waist         | -0.542630 | -0.160188 |
|   1 | left_shoulder      |  1.085436 |  2.104148 |
|   2 | left_elbow         | -1.948756 | -0.733276 |
|   3 | left_forearm_roll  | -0.146774 |  0.227906 |
|   4 | left_wrist_angle   |  0.857878 |  1.226435 |
|   5 | left_wrist_rotate  | -0.222097 |  0.130446 |
|   6 | left_gripper       | -0.005000 |  0.066800 |
|   7 | right_waist        |  0.128475 |  0.590061 |
|   8 | right_shoulder     |  1.178063 |  2.364325 |
|   9 | right_elbow        | -2.026086 | -0.788364 |
|  10 | right_forearm_roll | -0.048355 |  0.250269 |
|  11 | right_wrist_angle  |  0.872043 |  1.229174 |
|  12 | right_wrist_rotate | -0.214683 |  0.018421 |
|  13 | right_gripper      | -0.003100 |  0.061900 |

当前解释：

```text
action is absolute joint-position command.

action[t] is close to qpos[t+1].

Do not treat action as:
  - joint delta
  - qpos[t]
  - EEF pose
  - EEF delta
  - Cosmos3 dual-arm 20D action
```

---

## 4. 仍需确认的关键信息

以下信息必须继续追踪，因为它们会直接影响训练 label、模型输出解释和真机部署安全。

### 4.1 Gripper 单位

当前观测范围：

```text
left_gripper:  -0.0050 ~ 0.0668
right_gripper: -0.0031 ~ 0.0619
```

当前假设：

```text
可能是米制夹爪开口宽度，也可能是 driver 内部单位。
不能只靠数值猜，必须从采集代码或 driver wrapper 确认。
```

需要从哪里查：

```bash
grep -R "gripper" -n /project/peilab/wam/physical_WM | head -100
grep -R "gripper" -n /project/peilab/wam | head -100
grep -R "gripper_val" -n /project/peilab/wam | head -100
grep -R "gripper_val_mutiple" -n /project/peilab/wam | head -100
grep -R "joint_states" -n /project/peilab/wam | head -100
```

重点确认：

```text
1. /action[:, 6] 和 /action[:, 13] 是怎么写入的。
2. gripper 是开口宽度、joint7 position、归一化值，还是 driver 内部单位。
3. 有没有乘 2、除 1000、clip、normalize。
4. 负数是否是噪声、offset，还是允许值。
```

由什么决定：

```text
1. Piper driver / SDK 的 gripper command 表示。
2. 数据采集脚本是否做了单位转换。
3. 是否通过 ROS joint_states。
4. 是否使用 RViz 或其他控制 wrapper。
5. 是否存在 gripper scaling 参数。
```

会决定什么：

```text
1. gripper loss 如何计算。
2. gripper 是否单独归一化。
3. open / close 阈值。
4. safety filter 的 gripper min/max。
5. 真机部署时模型输出是否需要缩放。
6. gripper 是否允许轻微负数。
```

如果确认单位是 meter，安全配置可先写成：

```yaml
gripper:
  unit: meter
  min: 0.0
  max: 0.08
  max_delta_per_step: 0.005
```

如果确认单位是 normalized，安全配置应写成：

```yaml
gripper:
  unit: normalized
  min: 0.0
  max: 1.0
```

---

### 4.2 数据真实 FPS

当前 HDF5 只有 T，没有 timestamp 或 fps。

需要从哪里查：

```bash
find /project/peilab/wam -iname "*meta*" -o -iname "*.json" -o -iname "*.yaml" | head -100
grep -R "\"fps\"" -n /project/peilab/wam/physical_WM | head -50
grep -R "fps" -n /project/peilab/wam/physical_WM | head -50
grep -R "rate" -n /project/peilab/wam | head -100
grep -R "sleep" -n /project/peilab/wam | head -100
grep -R "frequency" -n /project/peilab/wam | head -100
grep -R "dt" -n /project/peilab/wam | head -100
```

优先查：

```text
1. LeRobot metadata
2. data collection script
3. camera config
4. robot control loop
5. converted video metadata
6.采集日志
```

如果有 mp4，可用：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate,avg_frame_rate,duration,nb_frames \
  -of default=noprint_wrappers=1:nokey=0 path/to/video.mp4
```

由什么决定：

```text
1. 相机采集频率。
2. robot state 读取频率。
3. action command 下发频率。
4. 保存数据时是否降采样。
5. LeRobot 转换时是否重新编码视频。
```

会决定什么：

```text
1. action_horizon=16 代表多久。
2. history_frames=2 代表多久。
3. policy_hz 应该是多少。
4. 真机部署时多久执行一次模型输出。
5. action[t] 到 qpos[t+1] 的真实延迟是多少秒。
6. rollout video 评估时的时间尺度。
```

例子：

```text
10 FPS: 16-step horizon = 1.6 秒
30 FPS: 16-step horizon = 0.53 秒
```

---

### 4.3 全数据 action-qpos 对齐

episode_0 已确认：

```text
action[t] ≈ qpos[t+1]
mean absolute difference ≈ 0.00095
```

但必须确认所有 episode 是否成立。

需要从哪里查：

```text
只能从数据本身计算。
```

需要比较：

```python
diff_same = abs(action - qpos)
diff_next = abs(action[:-1] - qpos[1:])
diff_prev = abs(action[1:] - qpos[:-1])
```

重点判断哪个最小。

如果 `diff_next` 最小，说明：

```text
obs[t] → action[t]
action[t] is next joint target
```

如果 `diff_same` 最小，说明：

```text
action[t] 可能接近当前状态，需要重新判断 action 是否是 command。
```

如果 `diff[t+k]` 最小，说明：

```text
控制或记录存在 k-frame delay。
```

由什么决定：

```text
1. 采集代码的记录顺序。
2. 控制器是否有延迟。
3. action command 是发送前记录，还是发送后记录。
4. qpos 是当前实际状态，还是目标状态。
```

会决定什么：

```text
1. 训练样本如何切。
2. target_action_index 是 t、t+1 还是 t+k。
3. 部署时模型输出如何解释。
4. 是否需要 latency compensation。
```

---

## 5. 其他需要继续确认的信息

### 5.1 数据和任务层面

| 信息                     | 从哪里查                                 | 决定什么                        |
| ---------------------- | ------------------------------------ | --------------------------- |
| 总 episode 数            | `find data -name "*.hdf5"`           | 训练规模、split                  |
| 是否只有 perfect           | `find data -type d`                  | 是否有失败数据                     |
| task instruction 来源    | LeRobot metadata / conversion script | language-conditioned policy |
| 每个 episode 是否同一任务      | metadata / 文件路径                      | 是否需要 task id                |
| 是否有 success label      | 路径 / metadata                        | 评估、筛选、失败学习                  |
| 是否有 timestamp          | HDF5 / metadata                      | 严格同步、延迟分析                   |
| 是否有 reset/start/end 标记 | metadata / logs                      | 去掉无效静止段                     |

### 5.2 图像层面

| 信息                    | 从哪里查             | 决定什么                |
| --------------------- | ---------------- | ------------------- |
| 三路相机是否固定安装            | 实验设置 / 图像检查      | 是否可跨 episode 共享视觉特征 |
| cam_high 位置           | 实验记录 / 外参        | 部署时相机必须复现           |
| wrist camera 左右是否对应正确 | 可视化 / 采集代码       | 避免左右臂混淆             |
| 是否有相机内参               | calibration 文件   | 后续 3D / EEF 研究      |
| 是否有相机外参               | calibration 文件   | 14D→20D EEF 研究      |
| 是否有掉帧                 | timestamp / 帧差检查 | 同步、数据清洗             |
| 是否有图像异常               | 可视化脚本            | 训练质量                |

### 5.3 机器人状态层面

| 信息                    | 从哪里查                      | 决定什么             |
| --------------------- | ------------------------- | ---------------- |
| qpos 单位               | driver / ROS joint_states | joint limit、安全执行 |
| qvel 单位               | driver / ROS joint_states | velocity limit   |
| effort 单位             | driver / CAN feedback     | 接触检测、失败分析        |
| 左右臂 joint 顺序          | 采集脚本 / URDF / driver      | action schema    |
| base_action 为什么全 0    | 采集代码                      | 是否忽略             |
| 是否有 robot mode/status | logs / driver             | 异常 episode 过滤    |

### 5.4 控制和部署层面

| 信息                | 从哪里查                    | 决定什么                                 |
| ----------------- | ----------------------- | ------------------------------------ |
| 真机控制接口            | SDK / ROS wrapper       | `send_joint_position_command` 怎么写    |
| 控制模式              | driver config           | joint position / velocity / MIT mode |
| command 频率        | 控制脚本                    | policy_hz、execute_steps              |
| 是否有插值器            | controller code         | action chunk 怎么执行                    |
| 急停接口              | driver / ROS service    | 部署安全                                 |
| enable/disable 流程 | driver / launch         | 真机启动脚本                               |
| gripper 控制接口      | driver / ROS service    | gripper command scaling              |
| 双臂 CAN 名称         | launch / udev / scripts | 左右臂不能接反                              |

### 5.5 训练层面

| 信息                      | 从哪里查           | 决定什么                       |
| ----------------------- | -------------- | -------------------------- |
| train/val/test split    | 自己生成           | 防止数据泄漏                     |
| action mean/std/min/max | train set 统计   | normalization              |
| qpos mean/std/min/max   | train set 统计   | qpos embedding             |
| gripper open/close 阈值   | 数据统计 + driver  | gripper loss / eval        |
| 静止段比例                   | action diff 统计 | 是否 trim                    |
| action 平滑度              | action diff 统计 | smoothness loss            |
| 每条 episode 长度分布         | scan_dataset   | batch sampling             |
| 多任务还是单任务                | metadata       | 是否需要 language conditioning |

---

## 6. 建议仓库文件结构

新增或整理：

```text
docs/
  PLAN_COSMOS3_PIPER_UPDATED.md
  ACTION_SCHEMA.md
  DATA_SCHEMA.md
  MISSING_METADATA_REPORT.md

configs/
  data/
    piper_dual_hdf5.yaml
  train/
    baseline_piper14.yaml
    cosmos3_piper14.yaml
  deploy/
    piper_dual_safety.yaml

scripts/
  check_action_qpos_alignment.py
  compute_dataset_stats.py
  visualize_episode.py
  split_dataset.py
  scan_missing_metadata.py

piper_cosmos/
  data/
    piper_dual_dataset.py
    hdf5_reader.py
  action/
    normalize.py
    safety_limits.py
  robot/
    safety_filter.py
    piper_client.py
  models/
    baseline_policy.py
    cosmos3_piper14_adapter.py
  eval/
    offline_action_eval.py
    safety_eval.py
```

---

## 7. 配置文件：`configs/data/piper_dual_hdf5.yaml`

Codex 创建该文件：

```yaml
dataset:
  name: piper_dual_pack_3_objects_plus
  root: /project/peilab/wam/physical_WM/data/pack_3_objects_plus
  default_split: perfect
  sim: false
  compress: false
  fps: UNKNOWN

hdf5:
  image_keys:
    cam_high: /observations/images/cam_high
    cam_left_wrist: /observations/images/cam_left_wrist
    cam_right_wrist: /observations/images/cam_right_wrist

  action_key: /action
  qpos_key: /observations/qpos
  qvel_key: /observations/qvel
  effort_key: /observations/effort
  base_action_key: /base_action

image:
  layout: THWC
  color: RGB
  dtype: uint8
  height: 480
  width: 640

action:
  dim: 14
  type: absolute_joint_position_command
  alignment: action_t_close_to_qpos_t_plus_1
  order:
    - left_waist
    - left_shoulder
    - left_elbow
    - left_forearm_roll
    - left_wrist_angle
    - left_wrist_rotate
    - left_gripper
    - right_waist
    - right_shoulder
    - right_elbow
    - right_forearm_roll
    - right_wrist_angle
    - right_wrist_rotate
    - right_gripper

observation:
  include_qpos: true
  include_qvel: false
  include_effort: false

training:
  observation_index: t
  target_action_index: t
  action_horizon: 16
  history_frames: 2
  stride: 1
  ignore_base_action: true

language:
  source: lerobot_metadata
  default_instruction: Put the three objects on the table into the container.

success:
  source: directory_name
  success_dirs:
    - perfect

missing_metadata:
  gripper_unit: UNKNOWN
  fps: UNKNOWN
  all_episode_action_qpos_alignment: UNKNOWN
```

---

## 8. Codex 任务 1：写 Action Schema 文档

创建：

```text
docs/ACTION_SCHEMA.md
```

必须包含：

```text
1. 14D action 是 absolute joint-position command。
2. action[t] 接近 qpos[t+1]。
3. 不要把 action 当 joint delta。
4. 不要把 action 当 EEF pose。
5. 不要第一阶段 padding 成 20D。
6. 每一维的名字和顺序。
7. gripper 单位仍需确认。
8. 部署时 action 输出作为下一步 joint target。
```

验收：

```text
docs/ACTION_SCHEMA.md 中必须明确：
"First-stage policy target is raw 14D absolute joint-position command."
```

---

## 9. Codex 任务 2：写 Missing Metadata 报告

创建：

```text
docs/MISSING_METADATA_REPORT.md
```

模板：

```markdown
# Missing Metadata Report

## Confirmed

- image layout: THWC
- image dtype: uint8
- image color: RGB
- action shape: [T,14]
- action type on episode_0: absolute joint-position command
- action[t] close to qpos[t+1] on episode_0

## Need to Confirm

### 1. Gripper unit

Current range:
- left_gripper: -0.0050 ~ 0.0668
- right_gripper: -0.0031 ~ 0.0619

Current hypothesis:
- likely opening width or driver internal gripper command

Sources to check:
- data collection script
- Piper SDK / ROS wrapper
- gripper command publish/service code

Impact:
- normalization
- safety limit
- deployment scaling
- gripper loss

### 2. FPS

Sources to check:
- LeRobot metadata
- data collection loop
- camera config
- converted video metadata
- logs

Impact:
- action horizon duration
- policy_hz
- control timing
- latency interpretation

### 3. Action-qpos alignment over all episodes

Command:
- scripts/check_action_qpos_alignment.py

Impact:
- target_action_index
- whether obs[t] -> action[t] is valid
- whether action is next-step joint target
```

---

## 10. Codex 任务 3：全数据检查 action/qpos 对齐

创建：

```text
scripts/check_action_qpos_alignment.py
```

运行：

```bash
python scripts/check_action_qpos_alignment.py \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect \
  --config configs/data/piper_dual_hdf5.yaml \
  --output reports/action_qpos_alignment_perfect.json
```

必须同时计算：

```python
diff_same = abs(action - qpos)
diff_next = abs(action[:-1] - qpos[1:])
diff_prev = abs(action[1:] - qpos[:-1])
```

输出：

```json
{
  "num_files": 0,
  "num_steps": 0,
  "best_alignment": "next",
  "same": {
    "global_mean_abs_diff": 0.0,
    "global_max_abs_diff": 0.0,
    "per_dim_mean_abs_diff": [],
    "per_dim_max_abs_diff": []
  },
  "next": {
    "global_mean_abs_diff": 0.0,
    "global_max_abs_diff": 0.0,
    "per_dim_mean_abs_diff": [],
    "per_dim_max_abs_diff": []
  },
  "prev": {
    "global_mean_abs_diff": 0.0,
    "global_max_abs_diff": 0.0,
    "per_dim_mean_abs_diff": [],
    "per_dim_max_abs_diff": []
  },
  "bad_files": []
}
```

判定标准：

```text
如果 next.global_mean_abs_diff < 0.005，且明显小于 same/prev，
则认为 action[t] ≈ qpos[t+1] 在全数据成立。
```

---

## 11. Codex 任务 4：扫描缺失 metadata

创建：

```text
scripts/scan_missing_metadata.py
```

运行：

```bash
python scripts/scan_missing_metadata.py \
  --repo-root /project/peilab/wam \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus \
  --output reports/missing_metadata_scan.md
```

脚本需要搜索：

```text
fps
rate
frequency
sleep
dt
gripper
gripper_val
gripper_val_mutiple
joint_states
cam_high
cam_left_wrist
cam_right_wrist
Put the three objects
success
perfect
```

输出：

```text
1. 可能包含 FPS 的文件和行号。
2. 可能包含 gripper 单位/缩放的文件和行号。
3. 可能包含 task instruction 的文件和行号。
4. 可能包含 success/perfect split 的文件和行号。
5. 没找到的项标记为 UNKNOWN。
```

---

## 12. Codex 任务 5：计算数据统计

创建：

```text
scripts/compute_dataset_stats.py
```

运行：

```bash
python scripts/compute_dataset_stats.py \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect \
  --config configs/data/piper_dual_hdf5.yaml \
  --output reports/dataset_stats_perfect.json
```

输出：

```json
{
  "num_files": 0,
  "num_steps": 0,

  "action_mean": [],
  "action_std": [],
  "action_min": [],
  "action_max": [],
  "action_p01": [],
  "action_p99": [],

  "qpos_mean": [],
  "qpos_std": [],
  "qpos_min": [],
  "qpos_max": [],

  "qvel_mean": [],
  "qvel_std": [],

  "gripper": {
    "left_min": 0.0,
    "left_max": 0.0,
    "right_min": 0.0,
    "right_max": 0.0
  },

  "episode_length": {
    "min": 0,
    "max": 0,
    "mean": 0.0
  }
}
```

要求：

```text
1. 当前 perfect 全量统计只用于数据理解。
2. 最终训练 normalization stats 必须只用 train split。
3. stats 后续训练和部署共用。
```

---

## 13. Codex 任务 6：写 Dataset Loader

创建：

```text
piper_cosmos/data/piper_dual_dataset.py
```

Dataset 返回：

```python
sample = {
    "images": {
        "cam_high": image_high,
        "cam_left_wrist": image_left,
        "cam_right_wrist": image_right,
    },
    "qpos": qpos_t,
    "action": action_chunk,
    "instruction": instruction,
    "episode_path": episode_path,
    "t": t,
}
```

shape 要求：

```text
cam_high: [history_frames, 3, H, W]
cam_left_wrist: [history_frames, 3, H, W]
cam_right_wrist: [history_frames, 3, H, W]
qpos: [14]
action: [action_horizon, 14]
instruction: str
```

默认参数：

```text
history_frames = 2
action_horizon = 16
image_size = 224 for baseline
image_size = 480 for Cosmos3
stride = 1
```

索引规则：

```text
valid t:
  t >= history_frames - 1
  t + action_horizon <= T
```

图像处理：

```text
原始: uint8 [H,W,3], RGB
输出: float32 [3,H,W]
范围: [0,1]
```

---

## 14. Codex 任务 7：可视化 episode

创建或更新：

```text
scripts/visualize_episode.py
```

运行：

```bash
python scripts/visualize_episode.py \
  --input /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect/episode_0.hdf5 \
  --config configs/data/piper_dual_hdf5.yaml \
  --output reports/episode_0_preview.mp4
```

视频要求：

```text
1. 横向拼接 cam_high、cam_left_wrist、cam_right_wrist。
2. overlay 当前 frame index。
3. overlay left_gripper 和 right_gripper。
4. 可选 overlay action dim 0-13 的简短数值。
5. 不做 BGR/RGB 反转。
```

验收：

```text
生成的视频颜色自然；
红色物体仍为红色；
三路相机时间同步肉眼合理。
```

---

## 15. Codex 任务 8：数据划分

创建：

```text
scripts/split_dataset.py
```

运行：

```bash
python scripts/split_dataset.py \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect \
  --output-dir splits/pack_3_objects_plus \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

输出：

```text
splits/pack_3_objects_plus/train.txt
splits/pack_3_objects_plus/val.txt
splits/pack_3_objects_plus/test.txt
```

要求：

```text
按 episode 文件划分，不按 frame 划分。
```

---

## 16. Codex 任务 9：训练 Baseline Policy

第一版先不用 Cosmos3，先验证数据闭环。

创建：

```text
piper_cosmos/models/baseline_policy.py
training/train_baseline_piper14.py
configs/train/baseline_piper14.yaml
```

模型输入：

```text
三路图像 + qpos
```

模型输出：

```text
[H, 14] absolute joint-position command
```

loss：

```text
L = action_mse + 2.0 * gripper_mse + 0.1 * smoothness_loss
```

运行：

```bash
python training/train_baseline_piper14.py \
  --config configs/train/baseline_piper14.yaml
```

验收：

```text
1. train loss 能下降。
2. val loss 不发散。
3. predicted action 曲线接近 ground truth。
4. gripper 预测不是常数。
```

---

## 17. Codex 任务 10：离线评估

创建：

```text
piper_cosmos/eval/offline_action_eval.py
```

评估指标：

```text
1. overall action MSE
2. per-dim MSE
3. left arm MSE
4. right arm MSE
5. left gripper MSE
6. right gripper MSE
7. action smoothness
8. out-of-range rate
```

输出：

```text
reports/offline_eval/
  metrics.json
  action_plots/
  pred_vs_gt_episode_*.png
```

进入部署前要求：

```text
1. action 预测曲线无明显高频抖动。
2. out-of-range rate 接近 0。
3. safety filter 离线通过率 > 99%。
```

---

## 18. Codex 任务 11：Safety Filter

创建：

```text
piper_cosmos/robot/safety_filter.py
configs/deploy/piper_dual_safety.yaml
```

输入：

```python
current_qpos: np.ndarray        # [14]
predicted_target: np.ndarray   # [14]
```

输出：

```python
safe_target: np.ndarray        # [14]
safety_info: dict
```

必须检查：

```text
1. NaN / Inf
2. joint target min/max
3. max delta from current_qpos
4. gripper range
5. action distribution p01/p99
6. optional velocity limit
```

第一版 safety config：

```yaml
action_type: absolute_joint_position

max_joint_delta_per_step: 0.02
max_gripper_delta_per_step: 0.005

use_dataset_range_limit: true
range_margin_ratio: 0.2

on_violation: hold_position

gripper:
  unit: UNKNOWN
  min: UNKNOWN
  max: UNKNOWN
```

伪代码：

```python
def filter_action(current_qpos, predicted_target, limits, stats):
    if has_nan_or_inf(predicted_target):
        return current_qpos, {"allowed": False, "reason": "nan_or_inf"}

    target = predicted_target.copy()

    target = clip_to_dataset_range(target, stats, margin_ratio=0.2)
    target = clip_to_robot_joint_limits(target, limits)

    delta = target - current_qpos

    delta[:6] = np.clip(delta[:6], -0.02, 0.02)
    delta[7:13] = np.clip(delta[7:13], -0.02, 0.02)

    delta[6] = np.clip(delta[6], -0.005, 0.005)
    delta[13] = np.clip(delta[13], -0.005, 0.005)

    safe_target = current_qpos + delta

    return safe_target, {"allowed": True, "reason": "ok"}
```

---

## 19. Codex 任务 12：Cosmos3-Piper14 Adapter

在 baseline 数据闭环通过后再做。

创建：

```text
piper_cosmos/models/cosmos3_piper14_adapter.py
configs/train/cosmos3_piper14.yaml
```

第一版目标：

```text
Cosmos3 visual/language backbone
+ qpos embedding
+ custom Piper14 action head
→ [H,14]
```

不要做：

```text
不要把 14D padding 成 20D。
不要假装 action 是 EEF delta。
不要丢掉 qpos。
```

训练目标：

```text
image history + instruction + qpos[t] → action[t:t+H]
```

最小验收：

```text
1. 能构造 batch。
2. 能跑 forward。
3. 能在 1-2 个 episode 上 overfit。
4. 输出 shape 为 [B, H, 14]。
```

---

## 20. Codex 任务 13：Policy Server 和 Shadow Mode

创建：

```text
piper_cosmos/models/policy_server.py
piper_cosmos/models/policy_client.py
tools/launch_shadow_mode.sh
```

Shadow mode 逻辑：

```text
1. 读取三路相机。
2. 读取当前 qpos。
3. 调用 policy。
4. 得到 action chunk。
5. 只运行 safety filter。
6. 不下发真机。
7. 保存日志。
```

日志：

```text
logs/shadow/YYYYMMDD_HHMMSS/
  config.yaml
  predicted_actions.npy
  current_qpos.npy
  safe_actions.npy
  safety_events.jsonl
  cam_high.mp4
  cam_left_wrist.mp4
  cam_right_wrist.mp4
```

验收：

```text
shadow mode 连续运行 30 分钟不崩溃；
safety violation rate < 0.5%；
不执行任何真机动作。
```

---

## 21. Codex 任务 14：低速真机部署

只有在 shadow mode 通过后执行。

创建：

```text
tools/launch_real_robot_slow.sh
piper_cosmos/robot/piper_client.py
```

部署循环：

```python
while True:
    images = read_three_cameras()
    qpos = read_qpos()
    action_chunk = policy(images, qpos, instruction)

    target = action_chunk[0]
    safe_target, info = safety_filter(qpos, target)

    if info["allowed"]:
        send_joint_position_command(safe_target)
    else:
        hold_position()
```

第一版限制：

```text
policy_hz = 2
execute_steps_per_prediction = 1
speed_scale = 0.1
max_joint_delta_per_step = 0.02 rad
max_gripper_delta_per_step = 0.005
```

真机任务顺序：

```text
1. hold position
2. 单臂小幅 reach
3. 单臂 gripper open/close
4. 单臂 grasp
5. 双臂小幅 reach
6. 双臂协作任务
```

---

## 22. 当前执行顺序

Codex 按顺序执行：

```text
1. docs/ACTION_SCHEMA.md
2. docs/MISSING_METADATA_REPORT.md
3. configs/data/piper_dual_hdf5.yaml
4. scripts/check_action_qpos_alignment.py
5. scripts/scan_missing_metadata.py
6. scripts/compute_dataset_stats.py
7. piper_cosmos/data/piper_dual_dataset.py
8. scripts/visualize_episode.py
9. scripts/split_dataset.py
10. baseline policy
11. offline_action_eval.py
12. safety_filter.py
13. Cosmos3-Piper14 adapter
14. shadow mode
15. real_robot_slow
```

不要跳过 1-12 直接做 Cosmos3。

---

## 23. 当前阻塞项

必须继续确认：

```text
P0:
1. 整个 perfect 数据目录里，是否所有 episode 都满足 action[t] ≈ qpos[t+1]。
2. 数据真实 FPS。
3. gripper command 的真实单位和安全上下限。

P1:
4. joint limits / max speed 写入 safety config。
5. task instruction 如何从 LeRobot metadata 映射到 HDF5。
6. train/val/test split。
7. action/qpos normalization stats。

P2:
8. 相机内参/外参。
9. effort 是否可用于接触检测。
10. 是否存在失败数据。
11. 静止段是否需要裁剪。
```

---

## 24. MVP 成功标准

最小可行版本完成条件：

```text
1. 能加载 HDF5 三路图像、qpos、action。
2. 能构造训练样本：
   images + qpos + instruction → action chunk [H,14]
3. 全数据 action-qpos alignment 已确认。
4. FPS 已确认。
5. gripper 单位或至少安全范围已确认。
6. Baseline policy loss 能下降。
7. 离线预测 action 曲线合理。
8. Safety filter 离线通过率 > 99%。
9. Shadow mode 可运行。
10. 真机低速执行单步 action 不危险。
```

MVP 完成后再进入：

```text
1. Cosmos3-Piper14 adapter fine-tuning。
2. 14D joint action → 20D EEF action conversion。
3. Forward dynamics / WAM 研究。
4. 双臂复杂协作任务。
```
