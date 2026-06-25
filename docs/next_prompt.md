# Next Prompt

Use this prompt for the next Codex session:

```text
You are continuing the Cosmos3 + Dual-Piper long-term research project.

Read only these files first:

1. docs/PLAN.md
2. docs/milestones/M2_dataset.md
3. docs/progress.md
4. docs/project_state.json
5. configs/data/piper_dual_hdf5.yaml
6. docs/ACTION_SCHEMA.md
7. docs/DATA_SCHEMA.md

Current status:

- M1 Metadata Infrastructure is complete.
- Do not start Baseline, Cosmos3, Safety Filter, or Deployment.
- Next phase is M2 Dataset Infrastructure planning/execution only.

Confirmed M1 metadata:

- Raw first-stage action is 14D absolute joint-position command.
- action[t] is closest to qpos[t+1] on the full perfect split.
- Formal report: reports/action_qpos_alignment_perfect.json
- num_files = 71
- num_steps = 65917
- best_alignment = next
- next.global_mean_abs_diff = 0.0009006630583410013
- bad_files = []
- Converted LeRobot metadata records fps = 30.
- Raw HDF5 has no fps attribute.
- Gripper command semantic is width.
- Existing deploy scripts clip gripper width commands to [0.0, 0.1].
- Dataset stats report: reports/dataset_stats_perfect.json
- Metadata scan report: reports/missing_metadata_scan.md

Mandatory constraints:

- Do not overwrite raw HDF5 files.
- Do not treat RGB as BGR.
- Do not treat /action as joint delta.
- Do not treat /action as EEF pose.
- Do not pad 14D action to 20D in the first stage.
- Do not drop qpos from policy input.
- Do not train a model until M2 dataset infrastructure is complete.
```
