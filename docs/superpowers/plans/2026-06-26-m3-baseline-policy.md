# M3 Baseline Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal offline baseline policy and smokeable training/evaluation scripts for the 14D Piper action chunk label.

**Architecture:** Implement a compact PyTorch `SimpleMultiViewCNNPolicy` that encodes three camera histories plus qpos and predicts `[B, 16, 14]`. Keep training and eval strictly offline, with short-run controls and graceful skip reports when PyTorch or compute resources are unavailable.

**Tech Stack:** Python, PyYAML, NumPy, optional PyTorch, existing `PiperDualDataset`.

---

### Task 1: Smoke Test First

**Files:**
- Create: `tests/test_m3_baseline.py`

- [ ] **Step 1: Add CLI/help and no-torch smoke behavior test**
- [ ] **Step 2: Run `pytest tests/test_m3_baseline.py -q` and confirm it fails because M3 files do not exist**

### Task 2: Baseline Policy Module

**Files:**
- Create: `piper_cosmos/models/__init__.py`
- Create: `piper_cosmos/models/baseline_policy.py`

- [ ] **Step 1: Implement `SimpleMultiViewCNNPolicy`**
- [ ] **Step 2: Implement `baseline_action_loss` with action MSE, 2.0 gripper MSE, and 0.1 smoothness loss**
- [ ] **Step 3: Keep imports py_compile-safe in environments without torch**

### Task 3: Training Script and Config

**Files:**
- Create: `configs/train/baseline_piper14.yaml`
- Create: `training/train_baseline_piper14.py`

- [ ] **Step 1: Load YAML config and build dataset/model when torch is available**
- [ ] **Step 2: Support `--max-steps`, `--limit-samples`, `--device`, `--smoke-report`, and `--dry-run`**
- [ ] **Step 3: If torch is missing, write a JSON smoke report with status `skipped`**

### Task 4: Offline Evaluation Script

**Files:**
- Create: `piper_cosmos/eval/__init__.py`
- Create: `piper_cosmos/eval/offline_action_eval.py`

- [ ] **Step 1: Add `--help` and config/checkpoint/report arguments**
- [ ] **Step 2: Compute action MSE, gripper MSE, smoothness, and out-of-range rate when torch/checkpoint are available**
- [ ] **Step 3: If torch is missing, write a JSON report with status `skipped`**

### Task 5: Verification and Docs

**Files:**
- Modify: `docs/progress.md`
- Modify: `docs/project_state.json`
- Modify: `docs/next_prompt.md`

- [ ] **Step 1: Run required help and py_compile commands**
- [ ] **Step 2: Run smoke test or skipped smoke report depending on torch availability**
- [ ] **Step 3: Update docs and stop before M4**
