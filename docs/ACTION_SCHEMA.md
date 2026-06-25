# Action Schema

## First-Stage Target

First-stage policy target is raw 14D absolute joint-position command.

The `/action` dataset must be interpreted as the next absolute joint target for the dual-Piper robot. Across the `perfect/` split, `action[t]` is closest to `qpos[t+1]`.

Do not interpret `/action` as:

- joint delta,
- current `qpos[t]`,
- EEF pose,
- EEF delta,
- Cosmos3 dual-arm 20D action.

Do not pad the 14D action to 20D in the first stage.

## Dimension Order

| Dim | Name |
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

## Deployment Interpretation

The policy output is interpreted as an absolute 14D joint target. Deployment should use:

```text
current_qpos
predicted absolute joint target
-> safety filter
-> safe joint target
-> joint-position command
```

## Known Unknowns

- Raw HDF5 files do not embed FPS as an attribute.

## Confirmed Gripper Interpretation

- Gripper dimensions: left dim 6, right dim 13.
- Unit/command semantic: gripper opening `width`.
- Deployment command range from existing robot scripts: clip to `[0.0, 0.1]` before `move_gripper(width=...)`.
- Dataset observed action range on `perfect/`: left `-0.0058 ~ 0.0807`, right `-0.0035 ~ 0.0738`.
- Values slightly below zero appear in recorded actions; deployment code clips commands to the non-negative width range.
