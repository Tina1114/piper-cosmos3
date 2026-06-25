# Missing Metadata Report

## Purpose

Track metadata gaps that affect labels, sampling, normalization, safety, and real-robot deployment. This file should be updated with evidence, not guesses.

## Confirmed

- image layout: THWC
- image dtype: uint8
- image color: RGB
- image resolution: 480 x 640
- cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- action shape: `[T, 14]`
- action type on `episode_0`: absolute joint-position command
- `action[t]` close to `qpos[t+1]` on `episode_0`
- raw HDF5 lacks `language`, `task`, and `success` keys

## Need to Confirm

### 1. Gripper Unit

Current range from inspected episode:

- left_gripper: `-0.0050 ~ 0.0668`
- right_gripper: `-0.0031 ~ 0.0619`

Current hypothesis:

- likely opening width or driver internal gripper command

Sources to check:

- data collection script
- Piper SDK or ROS wrapper
- gripper command publish/service code
- `joint_states` handling
- any gripper scaling parameter

Impact:

- normalization
- safety limit
- deployment scaling
- gripper loss
- open/close threshold

### 2. FPS

Sources to check:

- LeRobot metadata
- data collection loop
- camera config
- robot control loop
- converted video metadata
- logs

Impact:

- action horizon duration
- `policy_hz`
- control timing
- latency interpretation

### 3. Action-Qpos Alignment Over All Episodes

Status: confirmed for the `perfect/` split during Task 0 validation.

Command run:

```bash
python scripts/check_action_qpos_alignment.py \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect \
  --config configs/data/piper_dual_hdf5.yaml \
  --output /tmp/cosmos3_action_qpos_alignment_smoke.json
```

Result:

- `num_files`: 71
- `num_steps`: 65917
- `best_alignment`: `next`
- `next.global_mean_abs_diff`: `0.0009006630583410013`
- `next.global_max_abs_diff`: `0.049800001084804535`
- `bad_files`: `[]`

Interpretation:

- `action[t]` is closest to `qpos[t+1]` across the inspected `perfect/` split.
- Non-gripper joint dimensions are exactly aligned at `next` in this scan.
- Gripper dimensions still need unit and scaling confirmation.

Impact now resolved:

- `target_action_index`
- whether `obs[t] -> action[t]` is valid
- whether action is next-step joint target
- whether a one-frame next-target interpretation is valid

## Current Status

No full source-tree metadata scan has been run in this task. Run `scripts/scan_missing_metadata.py` in M1 follow-up work and paste evidence here.
