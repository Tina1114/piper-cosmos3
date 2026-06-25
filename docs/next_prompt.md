# Next Prompt

Use this prompt for the next Codex session:

```text
You are continuing the Cosmos3 + Dual-Piper long-term research project.

Read only these files first:

1. docs/PLAN.md
2. docs/milestones/M3_baseline.md
3. docs/progress.md
4. docs/project_state.json
5. configs/data/piper_dual_hdf5.yaml
6. configs/train/baseline_piper14.yaml
7. docs/ACTION_SCHEMA.md

Current status:

- M1 Metadata Infrastructure is complete.
- M2 Dataset Infrastructure is complete.
- M3 Baseline code is created.
- Do not start Cosmos3, Safety Filter, or Deployment.
- Do not start M4 until M3 forward/backward smoke is verified in a PyTorch environment.

M3 files:

- Model: piper_cosmos/models/baseline_policy.py
- Training script: training/train_baseline_piper14.py
- Training config: configs/train/baseline_piper14.yaml
- Offline eval: piper_cosmos/eval/offline_action_eval.py
- Smoke test: tests/test_m3_baseline.py

Baseline contract:

- Inputs: cam_high, cam_left_wrist, cam_right_wrist, qpos[14]
- Output: action chunk [16, 14]
- Target: obs[t] -> action[t:t+16]
- Loss: action_mse + 2.0 * gripper_mse + 0.1 * smoothness_loss

Local verification already run:

- python training/train_baseline_piper14.py --help
- python piper_cosmos/eval/offline_action_eval.py --help
- python -m py_compile piper_cosmos/models/baseline_policy.py training/train_baseline_piper14.py piper_cosmos/eval/offline_action_eval.py
- pytest tests/test_m3_baseline.py -q

Local smoke status:

- reports/baseline_piper14_smoke.json says skipped: torch_not_available
- reports/baseline_piper14_eval.json says skipped: torch_not_available
- Current Python environment has no torch, so forward/backward has not been verified.

Next task:

Run a 1-2 batch M3 smoke in an environment with PyTorch, preferably through SLURM if GPU is needed:

python training/train_baseline_piper14.py \
  --config configs/train/baseline_piper14.yaml \
  --max-steps 1 \
  --limit-samples 2 \
  --smoke-report reports/baseline_piper14_smoke.json

Mandatory constraints:

- Do not overwrite raw HDF5 files.
- Do not train long jobs on the login node.
- Do not use GPU outside SLURM.
- Do not create Cosmos3 adapter.
- Do not create Safety Filter.
- Do not write real-robot code.
```
