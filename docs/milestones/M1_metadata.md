# M1 Metadata Infrastructure

## Goal

Establish the minimum metadata, schema, configuration, and inspection tools needed before implementing a dataset loader or any training code.

## Scope

In scope:

- Compact project planning docs.
- Action and data schema docs.
- Missing metadata tracking.
- M1 HDF5 data config.
- CLI tools to check action/qpos alignment and scan likely metadata sources.

Out of scope:

- Dataset loader.
- Dataset statistics script beyond metadata checks.
- Visualization scripts.
- Train/val/test split generation.
- Baseline policy.
- Cosmos3 adapter.
- Safety filter implementation.
- Robot deployment code.

## Required Files

- `docs/PLAN.md`
- `docs/ACTION_SCHEMA.md`
- `docs/DATA_SCHEMA.md`
- `docs/MISSING_METADATA_REPORT.md`
- `docs/progress.md`
- `docs/next_prompt.md`
- `docs/project_state.json`
- `configs/data/piper_dual_hdf5.yaml`
- `scripts/check_action_qpos_alignment.py`
- `scripts/scan_missing_metadata.py`

## Confirmed Assumptions

- First-stage action dimension is 14.
- First-stage action type is absolute joint-position command.
- `action[t]` is expected to represent the next joint target and is close to `qpos[t+1]` on `episode_0`.
- Images are RGB, not BGR.
- `qpos` is required as model input.
- The default instruction is sourced from LeRobot metadata, not from the raw HDF5 file.

## Checks To Run

```bash
python scripts/check_action_qpos_alignment.py --help
python scripts/scan_missing_metadata.py --help
python -m py_compile scripts/check_action_qpos_alignment.py scripts/scan_missing_metadata.py
```

When data access is available:

```bash
python scripts/check_action_qpos_alignment.py \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect \
  --config configs/data/piper_dual_hdf5.yaml \
  --output reports/action_qpos_alignment_perfect.json
```

```bash
python scripts/scan_missing_metadata.py \
  --repo-root /project/peilab/wam \
  --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus \
  --output reports/missing_metadata_scan.md
```

## Exit Criteria

- Documentation is split into compact planning and milestone files.
- Project state can be restored from `PLAN.md`, this milestone file, `progress.md`, and `project_state.json`.
- The M1 config exists and encodes confirmed data schema plus unknown metadata fields.
- The M1 scripts have working `--help` output and pass Python syntax checks.
- Formal action/qpos alignment report exists under `reports/`.
- Metadata scan report exists under `reports/`.
- Dataset stats report exists under `reports/`.
- No Dataset Loader, training code, Cosmos3 code, or deployment code has been added.

## M1 Completion Status

- Status: complete.
- Formal `perfect/` split action/qpos alignment report: `reports/action_qpos_alignment_perfect.json`.
- Metadata scan report: `reports/missing_metadata_scan.md`.
- Dataset stats report: `reports/dataset_stats_perfect.json`.
- Converted dataset FPS: `30`.
- Gripper command semantic: opening `width`.
- Gripper deployment command range: `[0.0, 0.1]`.

Raw HDF5 caveat: raw episodes do not embed fps, task text, or per-step success labels.
