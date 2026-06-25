# Project Progress

## Current Milestone

M1 Metadata Infrastructure

## Completed In Task 0

- Read and reviewed `docs/PLAN_COSMOS3_PIPER_UPDATED.md`.
- Identified the original plan's organization issues:
  - long-term goals, current metadata tasks, dataset work, training, safety, and deployment were mixed in one file;
  - M1 metadata checks were interleaved with later Dataset Loader and model tasks;
  - confirmed facts and unknown metadata were repeated across several sections;
  - the execution order was useful but too long for repeated Codex context loading.
- Created compact `docs/PLAN.md` as the default project entry point.
- Split long-term work into milestone docs under `docs/milestones/`.
- Initialized action, data, and missing metadata docs.
- Created project structure directories:
  - `configs/`
  - `configs/data/`
  - `scripts/`
  - `reports/`
  - `piper_cosmos/`
- Created M1 config:
  - `configs/data/piper_dual_hdf5.yaml`
- Created M1 scripts:
  - `scripts/check_action_qpos_alignment.py`
  - `scripts/scan_missing_metadata.py`
- Ran alignment validation on the real `perfect/` split with output written to `/tmp`.
- Confirmed `best_alignment = next` across 71 files and 65,917 steps.
- Did not create Dataset Loader, training code, Cosmos3 adapter, safety filter, or deployment code.

## Remaining M1 Work

- Run metadata scan over `/project/peilab/wam` and the dataset root.
- Update `docs/MISSING_METADATA_REPORT.md` with evidence for:
  - FPS,
  - gripper unit and scaling,
  - task instruction source,
  - success/perfect split source.
- Decide how to record the confirmed `next` alignment in downstream dataset sampling.

## Blockers

- Real dataset FPS is unknown.
- Gripper unit and safe range are unknown.
- A linked worktree was created at `/tmp/cosmos3_cy-task0-docs-infra`, but file edits were applied to the provided workspace root because the patch tool operates there. Current dirty status is in `/project/peilab/wam/cosmos3_cy`.

## Verification Log

Passed:

```bash
python scripts/check_action_qpos_alignment.py --help
python scripts/scan_missing_metadata.py --help
python -m py_compile scripts/check_action_qpos_alignment.py scripts/scan_missing_metadata.py
python -c "import h5py; print('h5py ok')"
python -c "import yaml; print('yaml ok')"
python scripts/scan_missing_metadata.py --repo-root . --data-root . --output /tmp/cosmos3_missing_metadata_smoke.md --max-matches-per-file 2
python scripts/check_action_qpos_alignment.py --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect --config configs/data/piper_dual_hdf5.yaml --output /tmp/cosmos3_action_qpos_alignment_smoke.json
```

Alignment validation result:

- `num_files`: 71
- `num_steps`: 65917
- `best_alignment`: `next`
- `next.global_mean_abs_diff`: `0.0009006630583410013`
- `bad_files`: `[]`

Initial `/tmp` worktree validation failed before these commands because the files were not present there; actual files are in the workspace root.
