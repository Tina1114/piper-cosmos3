# 动作 Schema

## 第一阶段目标

第一阶段 policy goal 是原始 `14D` 绝对关节位置指令。

双臂 `/action` 应解释为下一时刻的绝对关节目标。`perfect/` split 全量检查支持：`action[t]` 最接近 `qpos[t+1]`。

不应将 `/action` 解释为：

- joint delta
- 当前 `qpos[t]`
- EEF 位姿
- EEF delta
- Cosmos3 dual-arm 20D action

第一阶段不允许将 14D action 填充为 20D。

## 维度顺序

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

## 部署解释

policy output 应按绝对 14D 目标使用：

```text
current_qpos
预测的绝对关节目标
→ safety filter
→ safe target
→ 关节位姿指令
```

## 已知缺口

- 原始 HDF5 未内嵌 FPS 属性。

## 已确认的夹爪理解

- 左右夹爪维度：6、13。
- 单位/语义：夹爪开口 `width`。
- 部署脚本约束：`move_gripper(width=...)` 前裁剪到 `[0.0, 0.1]`。
- `perfect/` 数据实测范围：
  - 左：`-0.0058 ~ 0.0807`
  - 右：`-0.0035 ~ 0.0738`
- `perfect/` 存在轻微负值；部署端已做非负裁剪。
