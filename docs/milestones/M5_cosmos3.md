# M5 Cosmos3-Piper14

## Goal

Adapt Cosmos3 / Cosmos3-Piper14 to the verified dual-Piper 14D joint-position interface.

## Scope

In scope:

- Cosmos3 visual/language backbone usage.
- `qpos` embedding.
- Piper14 action head.
- Forward and overfit checks.

Out of scope:

- 20D EEF action conversion.
- Real-robot deployment.

## Planned Files

- `configs/train/cosmos3_piper14.yaml`
- `piper_cosmos/models/cosmos3_piper14_adapter.py`

## Constraints

- Do not pad 14D action to 20D.
- Do not reinterpret the action as EEF delta.
- Do not remove `qpos` input.

## Exit Criteria

- Batch construction works.
- Forward pass works.
- Output shape is `[B, H, 14]`.
- Model can overfit one or two episodes.
