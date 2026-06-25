# M4 Safety Infrastructure

## Goal

Create and verify safety filtering before any real-robot execution path.

## Scope

In scope:

- Safety filter for 14D absolute joint-position targets.
- Safety config.
- Offline safety checks using dataset statistics and joint limits.

Out of scope:

- Policy server.
- Robot command client.
- Real execution.

## Planned Files

- `configs/deploy/piper_dual_safety.yaml`
- `piper_cosmos/robot/safety_filter.py`
- `piper_cosmos/eval/safety_eval.py`

## Required Checks

- NaN and Inf rejection.
- Joint target min/max.
- Max delta from current `qpos`.
- Gripper range and max delta.
- Dataset p01/p99 range limits with margin.
- Violation behavior: hold position.

## Exit Criteria

- Offline safety pass rate is greater than 99% on validation predictions.
- Every violation returns current position or another explicit hold target.
