# Data Schema

## Dataset

- Name: `piper_dual_pack_3_objects_plus`
- Root: `/project/peilab/wam/physical_WM/data/pack_3_objects_plus`
- Default split: `perfect`
- Sim data: `false`
- Compression: `false`
- FPS: `30` from converted LeRobot metadata; raw HDF5 has no fps attribute

## HDF5 Keys

| Field | HDF5 Key | Shape | Notes |
| ----- | -------- | ----- | ----- |
| action | `/action` | `[T, 14]` | absolute joint-position command |
| base_action | `/base_action` | `[T, 2]` | all zero in inspected example; ignored for M1 |
| qpos | `/observations/qpos` | `[T, 14]` | required policy input |
| qvel | `/observations/qvel` | `[T, 14]` | optional later input |
| effort | `/observations/effort` | `[T, 14]` | optional later analysis |
| cam_high | `/observations/images/cam_high` | `[T, 480, 640, 3]` | RGB |
| cam_left_wrist | `/observations/images/cam_left_wrist` | `[T, 480, 640, 3]` | RGB |
| cam_right_wrist | `/observations/images/cam_right_wrist` | `[T, 480, 640, 3]` | RGB |

## Image Schema

- Layout: THWC
- Dtype: `uint8`
- Color: RGB
- Height: 480
- Width: 640

## Language Schema

Raw HDF5 files do not contain language keys. The current default instruction comes from LeRobot metadata:

```text
Put the three objects on the table into the container.
```

## Success Schema

Raw HDF5 files do not contain a `success` key. Until better labels are found, the directory name `perfect` is treated as dataset-level success metadata.

## Metadata Gaps

- Timestamp availability is unknown.
- Camera intrinsics and extrinsics are unknown.

## Confirmed M1 Metadata

- Full `perfect/` split action/qpos alignment: `action[t]` is closest to `qpos[t+1]`.
- Gripper command semantic: opening `width`.
- Gripper deployment command range: `[0.0, 0.1]`.
