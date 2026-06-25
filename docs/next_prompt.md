# Next Prompt

Use this prompt for the next Codex session:

```text
You are continuing the Cosmos3 + Dual-Piper long-term research project.

Read only these files first:

1. docs/PLAN.md
2. docs/milestones/M1_metadata.md
3. docs/progress.md
4. docs/project_state.json

Current phase:

- M1 Metadata Infrastructure.
- Do not start Dataset Loader, Baseline, Cosmos3, Safety, or Deployment yet.
- Current goal is to finish missing metadata evidence for FPS and gripper unit.

Next task:

1. Run:

   python scripts/scan_missing_metadata.py \
     --repo-root /project/peilab/wam \
     --data-root /project/peilab/wam/physical_WM/data/pack_3_objects_plus \
     --output reports/missing_metadata_scan.md

2. Inspect `reports/missing_metadata_scan.md` for evidence about FPS, gripper unit/scaling, task instruction source, and success/perfect split.
3. Update docs/MISSING_METADATA_REPORT.md with evidence from the report.
4. Update docs/progress.md and docs/project_state.json.
5. Stop. Do not implement Dataset Loader or training code.

Already completed in Task 0:

- action/qpos alignment check over `/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect`
- `num_files = 71`
- `num_steps = 65917`
- `best_alignment = next`
- `next.global_mean_abs_diff = 0.0009006630583410013`
- `bad_files = []`

Mandatory constraints:

- Do not overwrite raw HDF5 files.
- Do not treat RGB as BGR.
- Do not treat /action as joint delta.
- Do not treat /action as EEF pose.
- Do not pad 14D action to 20D in the first stage.
- Do not drop qpos from policy input.
```
