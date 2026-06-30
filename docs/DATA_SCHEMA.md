# 数据 Schema

## 数据集信息

- 名称：`piper_dual_pack_3_objects_plus`
- 根目录：`/project/peilab/wam/physical_WM/data/pack_3_objects_plus`
- 默认 split：`perfect`
- sim 标记：`false`
- 压缩标记：`false`
- FPS：`30`（来自转换后的 LeRobot 元数据，原始 HDF5 无 fps 属性）

## HDF5 字段

| 字段 | HDF5 键 | 形状 | 说明 |
| ----- | -------- | ----- | ----- |
| action | `/action` | `[T, 14]` | 绝对关节位置指令 |
| base_action | `/base_action` | `[T, 2]` | 示例中全为 0，第一阶段忽略 |
| qpos | `/observations/qpos` | `[T, 14]` | policy 输入 |
| qvel | `/observations/qvel` | `[T, 14]` | 后续版本可选输入 |
| effort | `/observations/effort` | `[T, 14]` | 后续分析用途 |
| cam_high | `/observations/images/cam_high` | `[T, 480, 640, 3]` | RGB |
| cam_left_wrist | `/observations/images/cam_left_wrist` | `[T, 480, 640, 3]` | RGB |
| cam_right_wrist | `/observations/images/cam_right_wrist` | `[T, 480, 640, 3]` | RGB |

## 图像 Schema

- 布局：THWC
- 数据类型：`uint8`
- 颜色：RGB
- 高度：480
- 宽度：640

## 语言 Schema

原始 HDF5 无语言字段。当前默认指令来自 LeRobot 元数据（英文原文）：

```text
Put the three objects on the table into the container.
```

## 成功标签 Schema

原始 HDF5 无 `success` 字段。暂时以目录名 `perfect` 作为数据集级成功标签。

## 元数据缺口

- 时间戳是否存在：未知
- 摄像头内参/外参：未知

## 已确认元数据（M1）

- 全量 `perfect/` split 行为对齐确认：`action[t]` 与 `qpos[t+1]` 最匹配。
- 夹爪语义：开口 `width`。
- 夹爪命令部署约束：`[0.0, 0.1]`。
