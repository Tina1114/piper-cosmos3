#!/usr/bin/env python3
"""Offline baseline training smoke for 14D Piper action chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONFIG = Path("configs/train/baseline_piper14.yaml")
DEFAULT_SMOKE_REPORT = Path("reports/baseline_piper14_smoke.json")


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


def effective_device(requested: str) -> str:
    if requested.startswith("cuda") and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("cuda_requires_slurm")
    return requested


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


def subset_dataset(dataset: Any, limit_samples: int | None) -> Any:
    if limit_samples is None or limit_samples <= 0 or limit_samples >= len(dataset):
        return dataset
    from torch.utils.data import Subset

    return Subset(dataset, list(range(limit_samples)))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not torch_available():
        payload = {"status": "skipped", "reason": "torch_not_available"}
        write_json(args.smoke_report, payload)
        return payload

    import torch
    from torch.utils.data import DataLoader

    from piper_cosmos.data.piper_dual_dataset import PiperDualDataset
    from piper_cosmos.models.baseline_policy import baseline_action_loss, build_policy_from_config

    config = load_yaml(args.config)
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    if not isinstance(training_cfg, dict):
        training_cfg = {}

    device = effective_device(args.device or str(training_cfg.get("device", "cpu")))
    max_steps = int(args.max_steps if args.max_steps is not None else training_cfg.get("max_steps", 2))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 2))
    limit_samples = args.limit_samples
    if limit_samples is None:
        limit_samples = int(dataset_cfg.get("limit_samples", 0) or 0)
    learning_rate = float(training_cfg.get("learning_rate", 1.0e-4))

    dataset = PiperDualDataset(
        data_root=dataset_cfg.get("data_root", "/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect"),
        config_path=dataset_cfg.get("config_path", "configs/data/piper_dual_hdf5.yaml"),
        history_frames=int(dataset_cfg.get("history_frames", 2)),
        action_horizon=int(dataset_cfg.get("action_horizon", 16)),
        image_size=int(dataset_cfg.get("image_size", 224)),
        stride=int(dataset_cfg.get("stride", 1)),
    )
    dataset = subset_dataset(dataset, limit_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
    )

    model = build_policy_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[dict[str, float]] = []

    if args.dry_run:
        payload = {"status": "passed", "mode": "dry_run", "num_samples": len(dataset)}
        write_json(args.smoke_report, payload)
        return payload

    model.train()
    for step, batch in enumerate(loader, start=1):
        if step > max_steps:
            break
        images = {key: value.to(device) for key, value in batch["images"].items()}
        qpos = batch["qpos"].to(device)
        target = batch["action"].to(device)
        pred = model(images, qpos)
        loss, metrics = baseline_action_loss(
            pred,
            target,
            gripper_weight=float(training_cfg.get("gripper_weight", 2.0)),
            smoothness_weight=float(training_cfg.get("smoothness_weight", 0.1)),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics["step"] = float(step)
        losses.append(metrics)
        print(json.dumps(metrics))

    output_dir = Path(training_cfg.get("output_dir", "reports/baseline_piper14_debug"))
    checkpoint_path = output_dir / str(training_cfg.get("checkpoint_name", "baseline_piper14_debug.pt"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)

    payload = {
        "status": "passed",
        "mode": "train_smoke",
        "device": device,
        "num_steps": len(losses),
        "losses": losses,
        "checkpoint": str(checkpoint_path),
    }
    write_json(args.smoke_report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small offline Piper14 baseline policy.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training YAML config.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda device. CUDA requires SLURM.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit dataset samples for smoke runs.")
    parser.add_argument("--smoke-report", type=Path, default=DEFAULT_SMOKE_REPORT, help="JSON smoke report path.")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset/model but skip optimizer steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = run_training(args)
    except RuntimeError as exc:
        if str(exc) == "cuda_requires_slurm":
            payload = {"status": "skipped", "reason": "cuda_requires_slurm"}
            write_json(args.smoke_report, payload)
        else:
            raise
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
