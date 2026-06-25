#!/usr/bin/env python3
"""Compute numeric action/qpos/qvel statistics for a Piper HDF5 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ACTION_KEY = "/action"
DEFAULT_QPOS_KEY = "/observations/qpos"
DEFAULT_QVEL_KEY = "/observations/qvel"


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read --config.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return data


def hdf5_key(config: dict[str, Any], name: str, default: str) -> str:
    hdf5 = config.get("hdf5", {})
    if isinstance(hdf5, dict):
        value = hdf5.get(name, default)
        if isinstance(value, str):
            return value
    return default


def action_order(config: dict[str, Any]) -> list[str]:
    action = config.get("action", {})
    if isinstance(action, dict):
        order = action.get("order", [])
        if isinstance(order, list) and all(isinstance(item, str) for item in order):
            return order
    return []


def find_hdf5_files(data_root: Path) -> list[Path]:
    if data_root.is_file():
        return [data_root]
    files = sorted(data_root.rglob("*.hdf5")) + sorted(data_root.rglob("*.h5"))
    return sorted(set(files))


def as_float_list(values: np.ndarray) -> list[float]:
    return [float(x) for x in values.tolist()]


def summarize_array(array: np.ndarray, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": as_float_list(array.mean(axis=0)),
        f"{prefix}_std": as_float_list(array.std(axis=0)),
        f"{prefix}_min": as_float_list(array.min(axis=0)),
        f"{prefix}_max": as_float_list(array.max(axis=0)),
    }


def gripper_range(actions: np.ndarray, order: list[str], name: str) -> tuple[float | None, float | None]:
    if name not in order:
        return None, None
    idx = order.index(name)
    if idx >= actions.shape[1]:
        return None, None
    return float(actions[:, idx].min()), float(actions[:, idx].max())


def compute_stats(data_root: Path, config_path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required to read HDF5 files.") from exc

    config = load_config(config_path)
    action_key = hdf5_key(config, "action_key", DEFAULT_ACTION_KEY)
    qpos_key = hdf5_key(config, "qpos_key", DEFAULT_QPOS_KEY)
    qvel_key = hdf5_key(config, "qvel_key", DEFAULT_QVEL_KEY)

    action_chunks: list[np.ndarray] = []
    qpos_chunks: list[np.ndarray] = []
    qvel_chunks: list[np.ndarray] = []
    episode_lengths: list[int] = []
    bad_files: list[dict[str, str]] = []

    files = find_hdf5_files(data_root)
    for path in files:
        try:
            with h5py.File(path, "r") as h5:
                action = np.asarray(h5[action_key], dtype=np.float64)
                qpos = np.asarray(h5[qpos_key], dtype=np.float64)
                qvel = np.asarray(h5[qvel_key], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - keep scanning and report bad files.
            bad_files.append({"path": str(path), "error": str(exc)})
            continue

        if action.ndim != 2 or qpos.ndim != 2 or qvel.ndim != 2:
            bad_files.append({"path": str(path), "error": "expected 2D action/qpos/qvel arrays"})
            continue

        length = min(action.shape[0], qpos.shape[0], qvel.shape[0])
        if length <= 0:
            bad_files.append({"path": str(path), "error": "empty action/qpos/qvel array"})
            continue

        action_chunks.append(action[:length])
        qpos_chunks.append(qpos[:length])
        qvel_chunks.append(qvel[:length])
        episode_lengths.append(int(length))

    if not action_chunks:
        raise SystemExit(f"No readable HDF5 episodes found under {data_root}")

    actions = np.concatenate(action_chunks, axis=0)
    qpos = np.concatenate(qpos_chunks, axis=0)
    qvel = np.concatenate(qvel_chunks, axis=0)
    lengths = np.asarray(episode_lengths, dtype=np.float64)
    order = action_order(config)
    left_min, left_max = gripper_range(actions, order, "left_gripper")
    right_min, right_max = gripper_range(actions, order, "right_gripper")

    stats: dict[str, Any] = {
        "num_files": len(episode_lengths),
        "num_steps": int(actions.shape[0]),
        **summarize_array(actions, "action"),
        "action_p01": as_float_list(np.percentile(actions, 1, axis=0)),
        "action_p99": as_float_list(np.percentile(actions, 99, axis=0)),
        **summarize_array(qpos, "qpos"),
        **summarize_array(qvel, "qvel"),
        "episode_length_min": int(lengths.min()),
        "episode_length_max": int(lengths.max()),
        "episode_length_mean": float(lengths.mean()),
        "left_gripper_min": left_min,
        "left_gripper_max": left_max,
        "right_gripper_min": right_min,
        "right_gripper_max": right_max,
        "bad_files": bad_files,
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute action/qpos/qvel and episode-length statistics for HDF5 episodes."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="HDF5 file or directory.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config with HDF5 keys.")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_stats(args.data_root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
