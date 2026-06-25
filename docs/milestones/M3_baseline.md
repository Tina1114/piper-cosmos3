# M3 Baseline Policy

## Goal

Validate the data pipeline with a small non-Cosmos baseline policy before spending effort on Cosmos3 adaptation.

## Scope

In scope:

- Baseline image + qpos policy.
- Offline training loop.
- Offline action evaluation.
- Prediction plots and metrics.

Out of scope:

- Cosmos3 backbone integration.
- Real-robot execution.

## Planned Files

- `configs/train/baseline_piper14.yaml`
- `piper_cosmos/models/baseline_policy.py`
- `training/train_baseline_piper14.py`
- `piper_cosmos/eval/offline_action_eval.py`

## Exit Criteria

- Training loss decreases.
- Validation loss does not diverge.
- Predicted action curves are close to ground truth.
- Gripper predictions are not constant.
- Out-of-range rate is close to zero.

## Current Status

- Baseline code and CLI scaffolding have been created.
- Local `--help` and Python syntax checks pass.
- Local forward/backward smoke is blocked because the current Python environment does not have `torch`.
- GPU training must run through SLURM, not on the login node.
