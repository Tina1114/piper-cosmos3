#!/usr/bin/env python3
"""Offline action evaluation for the small Piper14 baseline policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONFIG = Path("configs/train/baseline_piper14.yaml")
DEFAULT_REPORT = Path("reports/baseline_piper14_eval.json")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    image_keys = samples[0]["images"].keys()
    images = {
        key: torch.as_tensor(
            [sample["images"][key] for sample in samples],
            dtype=torch.float32,
        )
        for key in image_keys
    }
    qpos = torch.as_tensor([sample["qpos"] for sample in samples], dtype=torch.float32)
    action = torch.as_tensor([sample["action"] for sample in samples], dtype=torch.float32)
    return {"images": images, "qpos": qpos, "action": action}


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if not torch_available():
        payload = {"status": "skipped", "reason": "torch_not_available"}
        write_json(args.report, payload)
        return payload

    import torch
    from torch.utils.data import DataLoader, Subset

    from piper_cosmos.data.piper_dual_dataset import PiperDualDataset
    from piper_cosmos.models.baseline_policy import (
        GRIPPER_DIMS,
        baseline_action_loss,
        build_policy_from_config,
    )

    config = load_yaml(args.config)
    dataset_cfg = config.get("dataset", {})
    eval_cfg = config.get("eval", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}

    dataset = PiperDualDataset(
        data_root=dataset_cfg.get("data_root", "/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect"),
        config_path=dataset_cfg.get("config_path", "configs/data/piper_dual_hdf5.yaml"),
        history_frames=int(dataset_cfg.get("history_frames", 2)),
        action_horizon=int(dataset_cfg.get("action_horizon", 16)),
        image_size=int(dataset_cfg.get("image_size", 224)),
        stride=int(dataset_cfg.get("stride", 1)),
    )
    limit_samples = args.limit_samples or int(dataset_cfg.get("limit_samples", 0) or 0)
    if limit_samples > 0 and limit_samples < len(dataset):
        dataset = Subset(dataset, list(range(limit_samples)))

    model = build_policy_from_config(config).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    max_batches = args.max_batches or int(eval_cfg.get("max_batches", 2))
    metrics: list[dict[str, float]] = []
    out_of_range = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if batch_idx > max_batches:
                break
            images = {key: value.to(args.device) for key, value in batch["images"].items()}
            qpos = batch["qpos"].to(args.device)
            target = batch["action"].to(args.device)
            pred = model(images, qpos)
            _, batch_metrics = baseline_action_loss(pred, target)
            metrics.append(batch_metrics)
            gripper = pred[..., list(GRIPPER_DIMS)]
            out_of_range.append(((gripper < 0.0) | (gripper > 0.1)).float().mean().item())

    if not metrics:
        raise SystemExit("No evaluation batches were produced.")

    keys = metrics[0].keys()
    summary = {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}
    summary["out_of_range_rate"] = sum(out_of_range) / len(out_of_range)
    payload = {
        "status": "passed",
        "checkpoint": str(args.checkpoint),
        "num_batches": len(metrics),
        "metrics": summary,
    }
    write_json(args.report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an offline Piper14 baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training/eval YAML config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Baseline checkpoint path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON report output path.")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum batches to evaluate.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit dataset samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_eval(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
