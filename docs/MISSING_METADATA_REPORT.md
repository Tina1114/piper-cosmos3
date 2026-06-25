# Missing Metadata Report

## Purpose

Track metadata gaps that affect labels, sampling, normalization, safety, and real-robot deployment. This file is evidence-based; do not fill unknown fields by guessing.

## Confirmed

- image layout: THWC
- image dtype: uint8
- image color: RGB
- image resolution: 480 x 640
- cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- action shape: `[T, 14]`
- action type: absolute joint-position command
- action alignment: `action[t]` is closest to `qpos[t+1]` on the full `perfect/` split
- dataset FPS for the converted LeRobot metadata: `30`
- gripper command semantic: opening `width`
- gripper deployment command range: `[0.0, 0.1]`
- default task instruction: `Put the three objects on the table into the container.`
- raw HDF5 lacks `language`, `task`, and `success` keys
- `perfect/` directory is treated as dataset-level success metadata

## Evidence From Task 1

Formal alignment report:

```bash
python scripts/check_action_qpos_alignment.py \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect \
  --config configs/data/piper_dual_hdf5.yaml \
  --output reports/action_qpos_alignment_perfect.json
```

Result:

- `num_files`: 71
- `num_steps`: 65917
- `best_alignment`: `next`
- `next.global_mean_abs_diff`: `0.0009006630583410013`
- `next.global_max_abs_diff`: `0.049800001084804535`
- `bad_files`: `[]`

Interpretation:

- `action[t]` is closest to `qpos[t+1]` across the `perfect/` split.
- Non-gripper joint dimensions are exactly aligned at `next` in this scan.
- Gripper dimensions differ slightly from next-step `qpos`, consistent with gripper command/state handling.

## Evidence From Task 2

Metadata scan report:

```bash
python scripts/scan_missing_metadata.py \
  --repo-root /project/peilab/wam \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus \
  --output reports/missing_metadata_scan.md
```

FPS evidence:

- `/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect_lerobot/meta/info.json` has `fps: 30`.
- The same `info.json` records `video.fps: 30` for `cam_high`, `cam_left_wrist`, and `cam_right_wrist`.
- `/project/peilab/wam/physical_WM/scripts/trian_scripts/convert_pack_3_objects_plus.sh` converts `perfect/` to `perfect_lerobot/` with `--fps 30`.
- The conversion script comments state that raw HDF5 has no fps attribute and that 30 is the conversion setting used for this dataset.

Gripper evidence:

- `/project/peilab/wam/physical_WM/scripts/deploy_real_bot/deploy_real_ckpt.py` clips dim 6 and dim 13 to `[0.0, 0.1]`.
- The same script calls `move_gripper(width=left_gripper_position, force=1.0)` and `move_gripper(width=right_gripper_position, force=1.0)`.
- `/project/peilab/wam/physical_WM/scripts/deploy_real_bot/deploy_real_ckpt_temporal_ensembling.py` uses the same clipping and `width=` command path.

Task/source evidence:

- `/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect_lerobot/meta/tasks.jsonl` stores `Put the three objects on the table into the container.`
- Raw HDF5 files still do not embed task text or success labels.

## Evidence From Task 3

Dataset stats report:

```bash
python scripts/compute_dataset_stats.py \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect \
  --config configs/data/piper_dual_hdf5.yaml \
  --output reports/dataset_stats_perfect.json
```

Result:

- `num_files`: 71
- `num_steps`: 65917
- `episode_length_min`: 738
- `episode_length_max`: 1244
- `episode_length_mean`: `928.4084507042254`
- `left_gripper_min`: `-0.005799999926239252`
- `left_gripper_max`: `0.08070000261068344`
- `right_gripper_min`: `-0.0035000001080334187`
- `right_gripper_max`: `0.0737999975681305`
- `bad_files`: `[]`

Interpretation:

- Observed gripper actions mostly sit inside the deployment command range, with slight negative values that should be clipped before robot command.
- Dataset stats are for metadata/schema understanding only and are not a Dataset Loader or training normalization implementation.

## Remaining Caveats

- Raw HDF5 files do not contain an fps attribute; `30` is confirmed from the converted LeRobot metadata and conversion script.
- Raw HDF5 files do not contain per-step `success` labels; `perfect/` remains dataset-level success metadata.
