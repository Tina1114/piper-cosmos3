# Project Progress

## Current Milestone

M1 Metadata Infrastructure

## Completed In Task 0

- Reorganized the long project plan into compact milestone docs.
- Initialized action/data/missing metadata docs.
- Created M1 config and scripts:
  - `configs/data/piper_dual_hdf5.yaml`
  - `scripts/check_action_qpos_alignment.py`
  - `scripts/scan_missing_metadata.py`
- Ran initial alignment validation on the real `perfect/` split with output written to `/tmp`.
- Confirmed `best_alignment = next` across 71 files and 65,917 steps.
- Did not create Dataset Loader, training code, Cosmos3 adapter, safety filter, or deployment code.

## Completed In Task 1-4

- Ran formal action/qpos alignment and saved:
  - `reports/action_qpos_alignment_perfect.json`
- Confirmed formal alignment results:
  - `num_files`: 71
  - `num_steps`: 65917
  - `best_alignment`: `next`
  - `next.global_mean_abs_diff`: `0.0009006630583410013`
  - `bad_files`: `[]`
- Improved and ran metadata scan, saved:
  - `reports/missing_metadata_scan.md`
- Confirmed converted dataset FPS:
  - `fps = 30` in `/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect_lerobot/meta/info.json`
  - conversion script uses `--fps 30`
  - raw HDF5 has no fps attribute
- Confirmed gripper command semantic and deployment command range:
  - dimensions 6 and 13 are gripper commands
  - existing deploy scripts call `move_gripper(width=...)`
  - existing deploy scripts clip gripper commands to `[0.0, 0.1]`
- Created and ran dataset stats script:
  - `scripts/compute_dataset_stats.py`
  - `reports/dataset_stats_perfect.json`
- Confirmed stats summary:
  - `num_files`: 71
  - `num_steps`: 65917
  - `episode_length_min`: 738
  - `episode_length_max`: 1244
  - `episode_length_mean`: `928.4084507042254`
  - `left_gripper_min`: `-0.005799999926239252`
  - `left_gripper_max`: `0.08070000261068344`
  - `right_gripper_min`: `-0.0035000001080334187`
  - `right_gripper_max`: `0.0737999975681305`
- Updated M1 docs and project state with evidence-based metadata.
- Did not create Dataset Loader, training code, Cosmos3 adapter, safety filter, deployment code, or copy HDF5 data.

## Remaining M1 Work

- None.

## Caveats

- Raw HDF5 files do not embed fps as an attribute; FPS is confirmed from converted LeRobot metadata and conversion scripts.
- Raw HDF5 files do not contain per-step `success` labels; `perfect/` remains dataset-level success metadata.

## Verification Log

Passed in Task 0:

```bash
python scripts/check_action_qpos_alignment.py --help
python scripts/scan_missing_metadata.py --help
python -m py_compile scripts/check_action_qpos_alignment.py scripts/scan_missing_metadata.py
python -c "import h5py; print('h5py ok')"
python -c "import yaml; print('yaml ok')"
python scripts/scan_missing_metadata.py --repo-root . --data-root . --output /tmp/cosmos3_missing_metadata_smoke.md --max-matches-per-file 2
python scripts/check_action_qpos_alignment.py --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect --config configs/data/piper_dual_hdf5.yaml --output /tmp/cosmos3_action_qpos_alignment_smoke.json
```

Passed in Task 1-4:

```bash
pytest tests/test_m1_metadata_scripts.py -q
python scripts/check_action_qpos_alignment.py --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect --config configs/data/piper_dual_hdf5.yaml --output reports/action_qpos_alignment_perfect.json
python scripts/scan_missing_metadata.py --repo-root /project/peilab/wam --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus --output reports/missing_metadata_scan.md
python scripts/compute_dataset_stats.py --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect --config configs/data/piper_dual_hdf5.yaml --output reports/dataset_stats_perfect.json
```
