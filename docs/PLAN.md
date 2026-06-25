# Cosmos3 + Dual-Piper Project Plan

This is the compact entry point for the long-term research project. Future Codex sessions should read this file plus the current milestone file, not the full archived planning document.

## Project Goal

Use dual-arm Piper real-robot demonstrations to build the data, training, safety, and deployment path for:

```text
Piper dual-arm real data
-> Cosmos3 / Cosmos3-Piper14 training
-> real-robot safety deployment
-> real-robot inference
```

The current phase is not model training. M1 metadata infrastructure is complete; the next phase is M2 dataset infrastructure.

## Current Route

First-stage MVP uses the confirmed raw 14D joint-position interface:

```text
3 RGB camera streams + qpos + instruction
-> policy
-> 14D absolute joint-position command
-> safety filter
-> dual Piper low-speed execution
```

20D EEF action conversion is a later research task and must not block the first-stage MVP.

## Non-Negotiable Constraints

1. Do not overwrite raw HDF5 files.
2. Do not treat RGB images as BGR.
3. Do not treat `/action` as joint delta.
4. Do not treat `/action` as EEF pose or EEF delta.
5. Do not pad 14D action to 20D in the first stage.
6. Do not drop `qpos` from policy input.
7. Real-robot scripts must default to shadow mode.
8. Real execution must require an explicit `--execute` style switch.
9. Safety filter failure must hold position.

## Confirmed Data Facts

- Example source: `/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect/episode_0.hdf5`
- HDF5 attrs: `compress = False`, `sim = False`
- Images: THWC, `uint8`, RGB, `480x640`
- Cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- `/action`: shape `[T, 14]`
- `/observations/qpos`: shape `[T, 14]`
- `/observations/qvel`: shape `[T, 14]`
- `/observations/effort`: shape `[T, 14]`
- `/base_action`: shape `[T, 2]`, all zero in the inspected example, ignored for first-stage work
- On `episode_0`, `action[t]` is close to `qpos[t+1]` with mean absolute difference around `0.00095`
- Default task instruction from LeRobot metadata: `Put the three objects on the table into the container.`
- The raw HDF5 does not contain `language`, `task`, or `success` keys.
- The `perfect/` directory can be treated as success demonstrations at dataset level until better labels are found.

## Current Status

Current milestone: [M1 Metadata Infrastructure](milestones/M1_metadata.md) complete.

Resolved M1 metadata:

- Converted LeRobot metadata records `fps = 30`; raw HDF5 has no fps attribute.
- Gripper command semantic is opening `width`.
- Existing deploy scripts clip gripper width commands to `[0.0, 0.1]`.

Confirmed alignment:

- Full `perfect/` split action/qpos alignment check covered 71 files and 65,917 steps.
- Best alignment is `next`: `action[t]` is closest to `qpos[t+1]`.
- `next.global_mean_abs_diff = 0.0009006630583410013`.
- No bad files were reported by the alignment script.

Generated M1 reports:

- `reports/action_qpos_alignment_perfect.json`
- `reports/missing_metadata_scan.md`
- `reports/dataset_stats_perfect.json`

## Milestone Index

1. [M1 Metadata Infrastructure](milestones/M1_metadata.md)
2. [M2 Dataset Infrastructure](milestones/M2_dataset.md)
3. [M3 Baseline Policy](milestones/M3_baseline.md)
4. [M4 Safety Infrastructure](milestones/M4_safety.md)
5. [M5 Cosmos3-Piper14](milestones/M5_cosmos3.md)
6. [M6 Real-Robot Deployment](milestones/M6_deployment.md)

## Source Archive

The original long planning document is retained for audit and historical context:

```text
docs/PLAN_COSMOS3_PIPER_UPDATED.md
```

Do not use the archived file as the default working context after this reorganization. Use `PLAN.md`, the active milestone, `progress.md`, and `project_state.json`.
