# Project Progress

## Current Milestone

M3 Baseline Policy

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

## Completed In M2 Dataset Infrastructure

- Created raw HDF5 reader:
  - `piper_cosmos/data/hdf5_reader.py`
- Created dataset wrapper:
  - `piper_cosmos/data/piper_dual_dataset.py`
- Created episode visualization CLI:
  - `scripts/visualize_episode.py`
- Created episode-level split CLI:
  - `scripts/split_dataset.py`
- Added M2 smoke test:
  - `tests/test_m2_dataset.py`
- Created M2 execution plan:
  - `docs/superpowers/plans/2026-06-26-m2-dataset-infrastructure.md`
- Generated M2 utility outputs:
  - `reports/pack_3_objects_plus_split.json`
  - `reports/episode_preview.png`
- Real data smoke result:
  - dataset length: 64,781 samples
  - image shapes: `(2, 3, 224, 224)` for all three cameras
  - qpos shape: `(14,)`
  - action shape: `(16, 14)`
  - split counts: train 57, val 7, test 7 episodes
- Did not create Baseline, Cosmos3 adapter, Safety Filter, deployment code, training code, or copy HDF5 data.

## Caveats

- Raw HDF5 files do not embed fps as an attribute; FPS is confirmed from converted LeRobot metadata and conversion scripts.
- Raw HDF5 files do not contain per-step `success` labels; `perfect/` remains dataset-level success metadata.
- Current login-node Python environment does not have `torch`; M3 forward/backward smoke was skipped locally and must be run in an environment with PyTorch, preferably via SLURM for GPU work.

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

Passed in M2:

```bash
pytest tests/test_m2_dataset.py -q
python scripts/visualize_episode.py --help
python scripts/split_dataset.py --help
python -m py_compile piper_cosmos/data/hdf5_reader.py piper_cosmos/data/piper_dual_dataset.py scripts/visualize_episode.py scripts/split_dataset.py
python -c "from piper_cosmos.data.piper_dual_dataset import PiperDualDataset; ds=PiperDualDataset('/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect','configs/data/piper_dual_hdf5.yaml'); s=ds[0]; print(len(ds)); print({k:v.shape for k,v in s['images'].items()}); print(s['qpos'].shape, s['action'].shape, s['instruction'], s['t'])"
python scripts/split_dataset.py --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect --output reports/pack_3_objects_plus_split.json --seed 0
python scripts/visualize_episode.py --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect --config configs/data/piper_dual_hdf5.yaml --output reports/episode_preview.png --t 1 --history-frames 2 --image-size 224
```

## Completed In M3 Baseline Policy

- Created M3 execution plan:
  - `docs/superpowers/plans/2026-06-26-m3-baseline-policy.md`
- Created baseline policy module:
  - `piper_cosmos/models/baseline_policy.py`
- Created training config:
  - `configs/train/baseline_piper14.yaml`
- Created offline training script:
  - `training/train_baseline_piper14.py`
- Created offline action eval script:
  - `piper_cosmos/eval/offline_action_eval.py`
- Added M3 smoke tests:
  - `tests/test_m3_baseline.py`
- Implemented loss contract:
  - `action_mse + 2.0 * gripper_mse + 0.1 * smoothness_loss`
- Added safeguards:
  - `--help` works without torch installed.
  - CUDA device requests are skipped outside SLURM.
  - Missing torch writes a JSON smoke/eval report instead of pretending training ran.
- Generated reports:
  - `reports/baseline_piper14_smoke.json`
  - `reports/baseline_piper14_eval.json`
- Did not create Cosmos3 adapter, Safety Filter, deployment code, or real-robot code.

M3 local smoke status:

- `reports/baseline_piper14_smoke.json`: `skipped`, reason `torch_not_available`
- `reports/baseline_piper14_eval.json`: `skipped`, reason `torch_not_available`
- Forward/backward was not run in this login-node Python environment.

Passed in M3:

```bash
pytest tests/test_m3_baseline.py -q
python training/train_baseline_piper14.py --help
python piper_cosmos/eval/offline_action_eval.py --help
python -m py_compile piper_cosmos/models/baseline_policy.py training/train_baseline_piper14.py piper_cosmos/eval/offline_action_eval.py
python training/train_baseline_piper14.py --config configs/train/baseline_piper14.yaml --max-steps 1 --limit-samples 2 --smoke-report reports/baseline_piper14_smoke.json
python piper_cosmos/eval/offline_action_eval.py --checkpoint reports/baseline_piper14_debug/baseline_piper14_debug.pt --report reports/baseline_piper14_eval.json
```
