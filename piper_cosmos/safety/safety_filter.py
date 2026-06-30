#!/usr/bin/env python3
"""Offline safety checks and diagnostics for 14D Piper absolute joint targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _np() -> Any:
    import numpy as np

    return np


GRIPPER_DIMS = (6, 13)


def load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return data


def load_safety_config(path: Path | str) -> dict[str, Any]:
    import yaml  # type: ignore

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Safety config must be a YAML mapping: {path}")
    return data


def load_dataset_stats(path: Path | str) -> dict[str, Any]:
    return load_json(path)


def _array(values: Any) -> Any:
    np = _np()
    return np.asarray(values, dtype=np.float32)


def _limits_from_stats(stats: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    np = _np()
    gripper_cfg = config.get("gripper", {}) if isinstance(config.get("gripper"), dict) else {}
    checks_cfg = config.get("checks", {}) if isinstance(config.get("checks"), dict) else {}

    joint_min = _array(stats["qpos_min"])
    joint_max = _array(stats["qpos_max"])
    gripper_min = float(gripper_cfg.get("gripper_min", gripper_cfg.get("min_width", 0.0)))
    gripper_max = float(gripper_cfg.get("gripper_max", gripper_cfg.get("max_width", 0.1)))
    joint_min = joint_min.copy()
    joint_max = joint_max.copy()
    joint_min[list(GRIPPER_DIMS)] = gripper_min
    joint_max[list(GRIPPER_DIMS)] = gripper_max

    action_p01 = _array(stats["action_p01"])
    action_p99 = _array(stats["action_p99"])
    range_margin_ratio = float(checks_cfg.get("dataset_margin_ratio", 0.2))
    margin = (action_p99 - action_p01) * range_margin_ratio
    return {
        "joint_min": joint_min,
        "joint_max": joint_max,
        "action_p01": action_p01,
        "action_p99": action_p99,
        "dataset_low": action_p01 - margin,
        "dataset_high": action_p99 + margin,
        "max_joint_delta_per_step": float(checks_cfg.get("max_joint_delta_per_step", 0.2)),
        "max_gripper_delta_per_step": float(checks_cfg.get("max_gripper_delta_per_step", 0.02)),
        "use_joint_limits": bool(checks_cfg.get("use_joint_limits", True)),
        "use_delta_limits": bool(checks_cfg.get("use_delta_limits", True)),
        "use_dataset_range_limits": bool(checks_cfg.get("use_dataset_range_limits", True)),
    }


def _finite_violations(target: Any) -> list[dict[str, Any]]:
    np = _np()
    violations: list[dict[str, Any]] = []
    for dim in np.where(~np.isfinite(target))[0].tolist():
        violations.append({"code": "non_finite", "dim": int(dim), "value": float(target[dim])})
    return violations


def evaluate_single_action(
    current_qpos: Any,
    target_action: Any,
    config: dict[str, Any],
    dataset_stats: dict[str, Any],
) -> dict[str, Any]:
    np = _np()
    current = _array(current_qpos)
    target = _array(target_action)
    limits = _limits_from_stats(dataset_stats, config)
    violations: list[dict[str, Any]] = _finite_violations(target)

    if limits["use_joint_limits"]:
        for dim in np.where(target < limits["joint_min"])[0].tolist():
            violations.append(
                {
                    "code": "joint_below_min" if dim not in GRIPPER_DIMS else "gripper_below_min",
                    "dim": int(dim),
                    "value": float(target[dim]),
                    "limit": float(limits["joint_min"][dim]),
                }
            )
        for dim in np.where(target > limits["joint_max"])[0].tolist():
            violations.append(
                {
                    "code": "joint_above_max" if dim not in GRIPPER_DIMS else "gripper_above_max",
                    "dim": int(dim),
                    "value": float(target[dim]),
                    "limit": float(limits["joint_max"][dim]),
                }
            )

    if limits["use_dataset_range_limits"]:
        for dim in np.where(target < limits["dataset_low"])[0].tolist():
            violations.append(
                {
                    "code": "dataset_below_p01_margin",
                    "dim": int(dim),
                    "value": float(target[dim]),
                    "limit": float(limits["dataset_low"][dim]),
                }
            )
        for dim in np.where(target > limits["dataset_high"])[0].tolist():
            violations.append(
                {
                    "code": "dataset_above_p99_margin",
                    "dim": int(dim),
                    "value": float(target[dim]),
                    "limit": float(limits["dataset_high"][dim]),
                }
            )

    if limits["use_delta_limits"]:
        delta = np.abs(target - current)
        joint_dims = [dim for dim in range(int(delta.shape[0])) if dim not in GRIPPER_DIMS]
        for dim in [joint_dims[idx] for idx in np.where(delta[joint_dims] > limits["max_joint_delta_per_step"])[0].tolist()]:
            violations.append(
                {
                    "code": "delta_above_max",
                    "dim": int(dim),
                    "value": float(delta[dim]),
                    "limit": float(limits["max_joint_delta_per_step"]),
                }
            )
        for dim in [GRIPPER_DIMS[idx] for idx in np.where(delta[list(GRIPPER_DIMS)] > limits["max_gripper_delta_per_step"])[0].tolist()]:
            violations.append(
                {
                    "code": "gripper_delta_above_max",
                    "dim": int(dim),
                    "value": float(delta[dim]),
                    "limit": float(limits["max_gripper_delta_per_step"]),
                }
            )

    accepted = len(violations) == 0
    safe_action = target if accepted else current
    return {
        "accepted": accepted,
        "safe_action": safe_action,
        "codes": sorted({item["code"] for item in violations}),
        "violations": violations,
    }


def evaluate_action_chunk(
    current_qpos: Any,
    action_chunk: Any,
    config: dict[str, Any],
    dataset_stats: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    np = _np()
    current = _array(current_qpos)
    chunk = _array(action_chunk)
    if chunk.ndim != 2:
        raise ValueError(f"action_chunk must have shape [H,D], got {tuple(chunk.shape)}")

    results: list[dict[str, Any]] = []
    q = current.copy()
    for horizon_idx in range(int(chunk.shape[0])):
        reference = current if mode in {"first_action", "chunk_against_initial"} else q
        evaluation = evaluate_single_action(reference, chunk[horizon_idx], config, dataset_stats)
        evaluation["horizon"] = horizon_idx
        results.append(evaluation)
        if mode == "rollout":
            q = _array(evaluation["safe_action"])
        if mode == "first_action":
            break
    return results
