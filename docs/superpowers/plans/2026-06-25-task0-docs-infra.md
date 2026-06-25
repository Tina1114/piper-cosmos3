# Task 0 Docs and Metadata Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the long Cosmos3 + dual-Piper project plan into a compact project index plus milestone documents, and initialize Milestone 1 metadata infrastructure without adding training, dataset loader, Cosmos3, or deployment code.

**Architecture:** `docs/PLAN.md` is the small entry point. Detailed execution content lives in `docs/milestones/`, while schema, progress, and state files support long-running AI-assisted project recovery. M1 includes only metadata configuration and two inspection scripts.

**Tech Stack:** Markdown docs, JSON project state, YAML configuration, Python CLI scripts using standard library plus optional `h5py`/`yaml` imports at runtime.

---

### Task 1: Reorganize Project Documentation

**Files:**
- Create: `docs/PLAN.md`
- Create: `docs/milestones/M1_metadata.md`
- Create: `docs/milestones/M2_dataset.md`
- Create: `docs/milestones/M3_baseline.md`
- Create: `docs/milestones/M4_safety.md`
- Create: `docs/milestones/M5_cosmos3.md`
- Create: `docs/milestones/M6_deployment.md`

- [x] **Step 1: Extract stable project facts from `docs/PLAN_COSMOS3_PIPER_UPDATED.md`**

Read the original plan and keep confirmed facts unchanged: raw HDF5 layout, RGB THWC image format, 14D absolute joint-position action, `action[t]` close to `qpos[t+1]` on `episode_0`, unknown FPS, unknown gripper unit, and mandatory first-stage safety principles.

- [x] **Step 2: Create compact `docs/PLAN.md`**

Include only project goal, route, current phase, non-negotiable constraints, current blockers, and milestone index.

- [x] **Step 3: Create milestone docs**

Split execution details so a future Codex session needs only `PLAN.md` plus the current milestone file.

### Task 2: Initialize Project Management Files

**Files:**
- Create: `docs/ACTION_SCHEMA.md`
- Create: `docs/DATA_SCHEMA.md`
- Create: `docs/MISSING_METADATA_REPORT.md`
- Create: `docs/progress.md`
- Create: `docs/next_prompt.md`
- Create: `docs/project_state.json`

- [x] **Step 1: Create action schema**

Document the 14D action order and explicitly state: `First-stage policy target is raw 14D absolute joint-position command.`

- [x] **Step 2: Create data schema**

Document HDF5 keys, image layout, observation keys, language source, success source, and unknown metadata.

- [x] **Step 3: Create missing metadata report**

Initialize gripper unit, FPS, and all-episode action/qpos alignment sections with sources to inspect and downstream impact.

- [x] **Step 4: Create progress and state files**

Record current milestone as M1 metadata infrastructure, completed Task 0 work, remaining work, blockers, and next task.

### Task 3: Create M1 Directories, Config, and Scripts

**Files:**
- Create directory: `configs/`
- Create directory: `configs/data/`
- Create directory: `scripts/`
- Create directory: `reports/`
- Create directory: `piper_cosmos/`
- Create: `configs/data/piper_dual_hdf5.yaml`
- Create: `scripts/check_action_qpos_alignment.py`
- Create: `scripts/scan_missing_metadata.py`

- [x] **Step 1: Create directory skeleton**

Only create requested infrastructure directories. Do not create training directories or dataset loader modules.

- [x] **Step 2: Create data config**

Encode confirmed HDF5 keys, image schema, action schema, default instruction, and unknown metadata fields.

- [x] **Step 3: Create alignment checker script**

Provide a CLI with `--data-root`, `--config`, and `--output`. Compute same, next, and prev action/qpos differences when `h5py` is available.

- [x] **Step 4: Create missing metadata scanner script**

Provide a CLI with `--repo-root`, `--data-root`, and `--output`. Search likely files for FPS, gripper, task instruction, success, and camera metadata terms.

### Task 4: Verify and Stop

**Files:**
- Modify: `docs/progress.md`
- Modify: `docs/next_prompt.md`
- Modify: `docs/project_state.json`

- [x] **Step 1: Run script help checks**

Run:

```bash
python scripts/check_action_qpos_alignment.py --help
python scripts/scan_missing_metadata.py --help
```

- [x] **Step 2: Run syntax checks**

Run:

```bash
python -m py_compile scripts/check_action_qpos_alignment.py scripts/scan_missing_metadata.py
```

- [x] **Step 3: Check git status**

Run:

```bash
git status --short
```

- [ ] **Step 4: Stop**

Report created/modified files, verification evidence, current blockers, and git status. Do not start Dataset, Baseline, Cosmos3, Safety, or Deployment work.
