#!/usr/bin/env python3
"""Check whether action[t] aligns with qpos[t], qpos[t+1], or qpos[t-1]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ACTION_KEY = "/action"
DEFAULT_QPOS_KEY = "/observations/qpos"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required when --config is provided. Install pyyaml or omit --config."
        ) from exc
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


def find_hdf5_files(data_root: Path) -> list[Path]:
    if data_root.is_file():
        return [data_root]
    files = sorted(data_root.rglob("*.hdf5")) + sorted(data_root.rglob("*.h5"))
    return sorted(set(files))


def init_stats(dim: int) -> dict[str, Any]:
    return {
        "count": 0,
        "sum_abs": [0.0] * dim,
        "max_abs": [0.0] * dim,
    }


def update_stats(stats: dict[str, Any], diff: Any) -> None:
    import numpy as np

    if diff.size == 0:
        return
    abs_diff = np.abs(diff)
    stats["count"] += int(abs_diff.shape[0])
    stats["sum_abs"] = (
        np.asarray(stats["sum_abs"], dtype=np.float64) + abs_diff.sum(axis=0)
    ).tolist()
    stats["max_abs"] = np.maximum(
        np.asarray(stats["max_abs"], dtype=np.float64), abs_diff.max(axis=0)
    ).tolist()


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats["count"])
    if count == 0:
        per_dim_mean = []
        global_mean = 0.0
    else:
        per_dim_mean = [float(x) / count for x in stats["sum_abs"]]
        global_mean = float(sum(stats["sum_abs"]) / (count * len(stats["sum_abs"])))
    per_dim_max = [float(x) for x in stats["max_abs"]]
    global_max = max(per_dim_max) if per_dim_max else 0.0
    return {
        "global_mean_abs_diff": global_mean,
        "global_max_abs_diff": float(global_max),
        "per_dim_mean_abs_diff": per_dim_mean,
        "per_dim_max_abs_diff": per_dim_max,
    }


def best_alignment(summary: dict[str, Any]) -> str:
    candidates = {
        name: summary[name]["global_mean_abs_diff"]
        for name in ("same", "next", "prev")
        if summary[name]["per_dim_mean_abs_diff"]
    }
    if not candidates:
        return "UNKNOWN"
    return min(candidates, key=candidates.get)


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required to read HDF5 files.") from exc

    config = load_config(args.config)
    action_key = hdf5_key(config, "action_key", DEFAULT_ACTION_KEY)
    qpos_key = hdf5_key(config, "qpos_key", DEFAULT_QPOS_KEY)
    files = find_hdf5_files(args.data_root)

    dim = int(args.action_dim)
    stats = {
        "same": init_stats(dim),
        "next": init_stats(dim),
        "prev": init_stats(dim),
    }
    bad_files: list[dict[str, str]] = []
    num_steps = 0

    for path in files:
        try:
            with h5py.File(path, "r") as h5:
                action = h5[action_key][:]
                qpos = h5[qpos_key][:]
        except Exception as exc:  # noqa: BLE001 - report file-specific failures.
            bad_files.append({"path": str(path), "error": str(exc)})
            continue

        if action.ndim != 2 or qpos.ndim != 2 or action.shape[1] != dim or qpos.shape[1] != dim:
            bad_files.append(
                {
                    "path": str(path),
                    "error": f"expected [T,{dim}] action/qpos, got {action.shape} and {qpos.shape}",
                }
            )
            continue

        length = min(action.shape[0], qpos.shape[0])
        action = action[:length]
        qpos = qpos[:length]
        num_steps += int(length)

        update_stats(stats["same"], action - qpos)
        update_stats(stats["next"], action[:-1] - qpos[1:])
        update_stats(stats["prev"], action[1:] - qpos[:-1])

    summary: dict[str, Any] = {
        "num_files": len(files) - len(bad_files),
        "num_steps": num_steps,
        "best_alignment": "UNKNOWN",
        "same": finalize_stats(stats["same"]),
        "next": finalize_stats(stats["next"]),
        "prev": finalize_stats(stats["prev"]),
        "bad_files": bad_files,
    }
    summary["best_alignment"] = best_alignment(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare /action against qpos at same, next, and previous timesteps."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="HDF5 file or directory.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config with HDF5 keys.")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
    parser.add_argument("--action-dim", type=int, default=14, help="Expected action/qpos dimension.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
