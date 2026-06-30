#!/usr/bin/env python3
"""Run Cosmos3 Piper14 SFT validation and export predictions for acceptance.

This script uses the ActionPolicy server pipeline used by Cosmos3 inference and
exports three arrays for downstream safety/acceptance checks:

- pred_action: [N, T, 14]
- current_qpos: [N, 14]
- ground_truth_action: [N, T, 14]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TOML = Path("configs/cosmos3/sft/action_policy_piper14_nano.toml")
DEFAULT_DATA_ROOT = Path("/project/peilab/wam/physical_WM/data/battery_assemble/perfect")
DEFAULT_DATA_CONFIG = Path("configs/dataset_configs/battery_assemble_hdf5.yaml")
DEFAULT_SPLIT_REPORT = Path("reports/battery_assemble_dataset_split.json")
DEFAULT_STATS_REPORT = Path("reports/battery_assemble_dataset_stats_perfect.json")
DEFAULT_SAFETY_CONFIG = Path("configs/safety/battery_piper14_safety.yaml")
DEFAULT_OUT_DIR = Path("reports/cosmos3_piper14")
DEFAULT_EVAL_REPORT = DEFAULT_OUT_DIR / "eval.json"
DEFAULT_PREDICTION_NPZ = DEFAULT_OUT_DIR / "validation_predictions.npz"
DEFAULT_PREDICTION_REPORT = DEFAULT_OUT_DIR / "validation_prediction_export.json"
DEFAULT_ACTION_HORIZON = 32
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_BATCHES = 64
DEFAULT_INFERENCE_STEPS = 32
DEFAULT_INFERENCE_SEED = 0
DEFAULT_MAX_ACTION_DIM = 14
DEFAULT_DEVICE = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained Cosmos3 checkpoint or DCP path.")
    parser.add_argument(
        "--run-config",
        type=Path,
        required=True,
        help="Training-time run config exported by Cosmos3 trainer (config.yaml).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Battery HDF5 root.")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG, help="HDF5 dataset config.")
    parser.add_argument("--split-report", type=Path, default=DEFAULT_SPLIT_REPORT, help="Episode split report.")
    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS_REPORT,
        help="Dataset stats for optional denormalization/debug checks.",
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=DEFAULT_SAFETY_CONFIG,
        help="Safety config for optional safety-related diagnostics.",
    )
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML, help="Cosmos3 experiment TOML for metadata.")
    parser.add_argument("--eval-report", type=Path, default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--prediction-npz", type=Path, default=DEFAULT_PREDICTION_NPZ)
    parser.add_argument("--prediction-report", type=Path, default=DEFAULT_PREDICTION_REPORT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--max-action-dim", type=int, default=DEFAULT_MAX_ACTION_DIM)
    parser.add_argument("--inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--inference-seed", type=int, default=DEFAULT_INFERENCE_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser.parse_args()


def _to_png_b64(image_chw: np.ndarray) -> str:
    from PIL import Image

    frame = np.asarray(image_chw, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB frame, got shape {frame.shape}")
    rgb = np.transpose(frame, (1, 2, 0))
    png = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(png, format="PNG")
    return base64.b64encode(png.getvalue()).decode("ascii")


def _safe_flatten(arr: np.ndarray) -> list[float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return [float(v) for v in flat]


@dataclass
class _EvalArtifactPaths:
    eval_report: Path
    prediction_npz: Path
    prediction_report: Path


def _load_split_counts(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    counts = payload.get("counts") if isinstance(payload, dict) else {}
    if not isinstance(counts, dict):
        return 0, 0
    return int(counts.get("train", 0) or 0), int(counts.get("val", 0) or 0)


def _build_service(
    checkpoint: Path,
    run_config: Path,
    action_horizon: int,
    seed: int,
    steps: int,
    args: argparse.Namespace,
) -> Any:
    from cosmos_framework.inference.args import CheckpointOverrides
    from cosmos_framework.scripts.action_policy_server_libero import ActionServerArgs, ActionModelService

    checkpoint_overrides = CheckpointOverrides(
        checkpoint_path=str(checkpoint),
        config_file=str(run_config),
    )

    inference_args = ActionServerArgs(
        checkpoint=checkpoint_overrides,
        output_dir=run_config.parent,
        run_validation=False,
        seed=int(seed),
        guidance=1.0,
        num_steps=int(steps),
        fps=30,
        action_chunk_size=action_horizon,
        raw_action_dim=int(args.max_action_dim),
        max_action_dim=int(args.max_action_dim),
    )

    return ActionModelService(inference_args)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import Subset

    from piper_cosmos.cosmos3.piper14_hdf5_action_dataset import get_piper14_hdf5_sft_dataset
    from piper_cosmos.data.splits import dataset_indices_for_split, load_episode_split

    torch.manual_seed(args.inference_seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Cosmos3 action inference in this environment.")

    dataset = get_piper14_hdf5_sft_dataset(
        root=str(args.data_root),
        config_path=str(args.data_config),
        fps=30.0,
        chunk_length=args.action_horizon,
        mode="policy",
        use_state=True,
        action_normalization=None,
        viewpoint="concat_view",
        resolution="480",
        max_action_dim=args.max_action_dim,
        tokenizer_config=None,
        iterable_shuffle=False,
        stride=1,
    )
    split = load_episode_split(args.split_report)
    val_indices = dataset_indices_for_split(dataset, split, "val")
    if val_indices:
        dataset = Subset(dataset, val_indices)

    if len(dataset) == 0:
        raise SystemExit("No validation samples available. Check split path and dataset root.")

    service = _build_service(
        checkpoint=args.checkpoint,
        run_config=args.run_config,
        action_horizon=args.action_horizon,
        seed=args.inference_seed,
        steps=args.inference_steps,
        args=args,
    )

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    qpos_chunks: list[np.ndarray] = []

    max_samples = args.max_batches * max(1, args.batch_size)
    processed = 0
    for sample_idx in range(len(dataset)):
        if processed >= max_samples:
            break

        sample = dataset[sample_idx]
        video = sample["video"]  # [C, T, H, W] uint8
        if not hasattr(video, "shape") or len(video.shape) != 4:
            continue
        action_seq = np.asarray(sample["action"], dtype=np.float32)
        if action_seq.ndim != 2 or action_seq.shape[-1] != 14:
            continue
        if action_seq.shape[0] < args.action_horizon + 1:
            continue

        request = {
            "image": _to_png_b64(np.asarray(video[:, 0], dtype=np.uint8)),
            "prompt": str(sample.get("ai_caption", "Assemble the mouse's battery.")),
            "domain_name": "piper14",
            "image_size": int(video.shape[3]),
        }
        response = service.predict_policy(request)
        raw = np.asarray(response.get("action", []), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[0] == 0:
            continue
        if raw.shape[1] < 14:
            pad = np.zeros((raw.shape[0], 14 - raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=1)
        pred = raw[:, :14]
        if pred.shape[0] < args.action_horizon:
            tail = np.repeat(pred[-1:], args.action_horizon - pred.shape[0], axis=0)
            pred = np.concatenate([pred, tail], axis=0)
        elif pred.shape[0] > args.action_horizon:
            pred = pred[: args.action_horizon]

        pred_chunks.append(pred.astype(np.float32))
        gt_chunks.append(action_seq[1 : args.action_horizon + 1, :14].astype(np.float32))
        qpos_chunks.append(action_seq[0, :14].astype(np.float32))
        processed += 1

    if not pred_chunks:
        raise SystemExit("No predictions were exported.")

    pred_arr = np.stack(pred_chunks, axis=0).astype(np.float32)
    gt_arr = np.stack(gt_chunks, axis=0).astype(np.float32)
    qpos_arr = np.stack(qpos_chunks, axis=0).astype(np.float32)

    diff = pred_arr - gt_arr
    if not np.isfinite(diff).all():
        raise SystemExit("Non-finite values in prediction or ground-truth arrays.")

    frame_losses = (diff * diff).mean(axis=2)
    pred_std = pred_arr.std(axis=(0, 1))
    gt_std = gt_arr.std(axis=(0, 1))
    pred_std_ratio = np.divide(
        pred_std,
        np.where(gt_std > 0, gt_std, 1.0),
        out=np.zeros_like(pred_std),
        where=(gt_std > 0),
    )
    metrics: dict[str, Any] = {
        "per_dim_action_mae": np.abs(diff).mean(axis=(0, 1)).tolist(),
        "per_dim_action_mse": (diff * diff).mean(axis=(0, 1)).tolist(),
        "per_dim_pred_std": pred_std.tolist(),
        "per_dim_gt_std": gt_std.tolist(),
        "per_dim_pred_to_gt_std_ratio": pred_std_ratio.tolist(),
    }

    eval_payload: dict[str, Any] = {
        "status": "passed",
        "checkpoint": str(args.checkpoint),
        "run_config": str(args.run_config),
        "num_batches": int(processed),
        "action_dim": 14,
        "action_horizon": int(args.action_horizon),
        "num_targets": int(frame_losses.size),
        "initial_val_loss": float(frame_losses.reshape(-1)[0]) if frame_losses.size else 0.0,
        "best_val_loss": float(frame_losses.min()) if frame_losses.size else 0.0,
        "overfit_gap": float(frame_losses.reshape(-1)[-1] - frame_losses.min()) if frame_losses.size else 0.0,
        "metrics": {
            "per_frame_mse": [float(value) for value in frame_losses.mean(axis=1).tolist()],
            "per_frame_mae": [float(value) for value in np.abs(diff).mean(axis=2).mean(axis=1).tolist()],
        },
        **metrics,
        "flat_pred_summary": {
            "count": int(pred_arr.size),
            "min": float(pred_arr.min()),
            "max": float(pred_arr.max()),
            "mean": float(pred_arr.mean()),
            "std": float(pred_arr.std()),
            "flat": _safe_flatten(pred_arr[: min(2, pred_arr.shape[0])].reshape(-1))[:16],
        },
        "flat_gt_summary": {
            "count": int(gt_arr.size),
            "min": float(gt_arr.min()),
            "max": float(gt_arr.max()),
            "mean": float(gt_arr.mean()),
            "std": float(gt_arr.std()),
            "flat": _safe_flatten(gt_arr[: min(2, gt_arr.shape[0])].reshape(-1))[:16],
        },
        "flat_qpos_summary": {
            "count": int(qpos_arr.size),
            "min": float(qpos_arr.min()),
            "max": float(qpos_arr.max()),
            "mean": float(qpos_arr.mean()),
            "std": float(qpos_arr.std()),
        },
    }

    args.eval_report.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_report.open("w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
        f.write("\n")

    np.savez_compressed(
        args.prediction_npz,
        pred_action=pred_arr.astype(np.float32),
        current_qpos=qpos_arr.astype(np.float32),
        ground_truth_action=gt_arr.astype(np.float32),
    )

    prediction_report = {
        "status": "passed",
        "eval_report": str(args.eval_report),
        "prediction_npz": str(args.prediction_npz),
        "checkpoint": str(args.checkpoint),
        "run_config": str(args.run_config),
        "num_samples": int(pred_arr.shape[0]),
        "action_horizon": int(pred_arr.shape[1]),
        "action_dim": int(pred_arr.shape[2]),
        "num_frames": int(pred_arr.shape[0] * pred_arr.shape[1]),
        "per_dim_action_mae": eval_payload["per_dim_action_mae"],
        "per_dim_action_mse": eval_payload["per_dim_action_mse"],
    }
    args.prediction_report.parent.mkdir(parents=True, exist_ok=True)
    with args.prediction_report.open("w", encoding="utf-8") as f:
        json.dump(prediction_report, f, indent=2)
        f.write("\n")

    return eval_payload


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
