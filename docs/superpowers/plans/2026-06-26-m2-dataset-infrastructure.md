# M2 Dataset Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the M2 dataset infrastructure for reading dual-Piper raw HDF5 episodes without training, deployment, or Cosmos3 integration.

**Architecture:** Keep raw HDF5 access in `piper_cosmos/data/hdf5_reader.py`, sampling/window logic in `piper_cosmos/data/piper_dual_dataset.py`, and operator utilities in `scripts/visualize_episode.py` and `scripts/split_dataset.py`. Return NumPy float32 arrays because PyTorch is not available in the current environment.

**Tech Stack:** Python, h5py, NumPy, PyYAML, Pillow, pytest.

---

### Task 1: Dataset Smoke Test

**Files:**
- Create: `tests/test_m2_dataset.py`

- [ ] **Step 1: Write a failing smoke test**

Create a temporary HDF5 episode with three RGB cameras, qpos, qvel, and action. Assert that `PiperDualDataset` returns:

- `images` with keys `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- image histories shaped `[2, 3, 224, 224]`
- `qpos` shaped `[14]`
- `action` shaped `[16, 14]`
- instruction, episode path, and timestep metadata

Run: `pytest tests/test_m2_dataset.py -q`
Expected: FAIL because `piper_cosmos.data.piper_dual_dataset` does not exist.

### Task 2: HDF5 Reader

**Files:**
- Create: `piper_cosmos/data/__init__.py`
- Create: `piper_cosmos/data/hdf5_reader.py`

- [ ] **Step 1: Implement config loading and episode discovery**
- [ ] **Step 2: Implement `HDF5EpisodeReader` methods for length, qpos, qvel, action chunks, image histories, and summary metadata**
- [ ] **Step 3: Ensure images are RGB THWC input converted to float32 CHW histories in `[0, 1]`**

Run: `python -m py_compile piper_cosmos/data/hdf5_reader.py`
Expected: exit 0.

### Task 3: Dataset Wrapper

**Files:**
- Create: `piper_cosmos/data/piper_dual_dataset.py`

- [ ] **Step 1: Build episode-level index using `history_frames`, `action_horizon`, and `stride`**
- [ ] **Step 2: Return the exact sample dict requested by the M2 prompt**
- [ ] **Step 3: Keep action target at `action[t:t+action_horizon]`, matching M1 `obs[t] -> action[t]` semantics**

Run: `pytest tests/test_m2_dataset.py -q`
Expected: PASS.

### Task 4: Utility Scripts

**Files:**
- Create: `scripts/visualize_episode.py`
- Create: `scripts/split_dataset.py`

- [ ] **Step 1: Add `visualize_episode.py` CLI with `--help`, episode selection, camera grid output, and no source HDF5 writes**
- [ ] **Step 2: Add `split_dataset.py` CLI with deterministic episode-level train/val/test split JSON**

Run:

```bash
python scripts/visualize_episode.py --help
python scripts/split_dataset.py --help
```

Expected: both exit 0.

### Task 5: Documentation and Verification

**Files:**
- Modify: `docs/progress.md`
- Modify: `docs/project_state.json`
- Modify: `docs/next_prompt.md`

- [ ] **Step 1: Update M2 status without entering M3**
- [ ] **Step 2: Run required verification**

Run:

```bash
python scripts/visualize_episode.py --help
python scripts/split_dataset.py --help
python -m py_compile \
  piper_cosmos/data/hdf5_reader.py \
  piper_cosmos/data/piper_dual_dataset.py \
  scripts/visualize_episode.py \
  scripts/split_dataset.py
pytest tests/test_m2_dataset.py -q
python -m json.tool docs/project_state.json
git status
```

Expected: all verification commands exit 0 except `git status`, which reports the changed files.
