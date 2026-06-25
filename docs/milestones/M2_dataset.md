# M2 Dataset Infrastructure

## Goal

Build the actual dataset infrastructure after M1 metadata checks are complete.

## Scope

In scope:

- Dataset statistics over HDF5 episodes.
- Episode visualization.
- Episode-level train/val/test split.
- HDF5 reader and dataset loader for three RGB cameras, `qpos`, instruction, and 14D action chunks.

Out of scope:

- Policy training.
- Cosmos3 integration.
- Real-robot deployment.

## Planned Files

- `scripts/compute_dataset_stats.py`
- `scripts/visualize_episode.py`
- `scripts/split_dataset.py`
- `piper_cosmos/data/hdf5_reader.py`
- `piper_cosmos/data/piper_dual_dataset.py`

## Entry Conditions

- M1 confirms all-episode action/qpos alignment or records the required target offset.
- FPS is known or explicitly handled as unknown in sampling decisions.
- Gripper unit or at least safe numeric range is documented.

## Exit Criteria

- Dataset loader returns:
  - three RGB camera histories as float tensors,
  - `qpos[t]`,
  - instruction string,
  - action chunk `[H, 14]`,
  - episode path and timestep.
- Train/val/test splits are episode-level, not frame-level.
- Statistics needed for normalization are computed from train split only for training use.
