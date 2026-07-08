#!/usr/bin/env python3
"""Batch-export fixed battery Cosmos3 checkpoints to inference-only HF dirs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO_ROOT / "reports/cosmos3_piper14/cosmos3_action/battery_piper14/battery_piper14_cosmos3_nano_20000step_4gpu_b8_acc1_offline"
CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"
CONFIG_FILE = RUN_ROOT / "config.yaml"
QWEN_SNAPSHOT = REPO_ROOT / "external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
OUTPUT_ROOT = REPO_ROOT / "cosmos_battery"
DEFAULT_STEPS = (12000, 16000, 18000, 20000)


def checkpoint_path_for_step(step: int) -> Path:
    return CHECKPOINT_ROOT / f"iter_{step:09d}" / "model"


def output_dir_for_step(step: int, output_root: Path) -> Path:
    return output_root / f"cosmos3_battery_piper14_step{step:05d}_hf_inference_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="Checkpoint steps to export. Default: %(default)s",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory under which exported HF model dirs will be created.",
    )
    parser.add_argument(
        "--exporter",
        type=Path,
        default=REPO_ROOT / "scripts/export_cosmos3_model_local.py",
        help="Path to the single-checkpoint exporter.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to invoke the exporter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow exporting into an existing non-empty destination directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exports without executing them.",
    )
    return parser.parse_args()


def validate_inputs(steps: Iterable[int], exporter: Path, output_root: Path) -> None:
    missing = []
    if not exporter.is_file():
        missing.append(f"missing exporter: {exporter}")
    if not CONFIG_FILE.is_file():
        missing.append(f"missing config: {CONFIG_FILE}")
    if not QWEN_SNAPSHOT.is_dir():
        missing.append(f"missing Qwen snapshot: {QWEN_SNAPSHOT}")
    for step in steps:
        checkpoint_path = checkpoint_path_for_step(step)
        if not checkpoint_path.is_dir():
            missing.append(f"missing checkpoint for step {step}: {checkpoint_path}")
    if missing:
        raise FileNotFoundError("\n".join(missing))
    output_root.mkdir(parents=True, exist_ok=True)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(REPO_ROOT / "external/cosmos/checkpoints/hf_home"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("COSMOS3_QWEN_SNAPSHOT", str(QWEN_SNAPSHOT))
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [
        str(REPO_ROOT),
        str(REPO_ROOT / "external/cosmos/packages/cosmos3"),
    ]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


def export_step(step: int, args: argparse.Namespace, env: dict[str, str]) -> None:
    checkpoint_path = checkpoint_path_for_step(step)
    output_dir = output_dir_for_step(step, args.output_root)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Refusing to export into non-empty directory without --force: {output_dir}"
        )
    command = [
        str(args.python_bin),
        str(args.exporter),
        "--checkpoint-path",
        str(checkpoint_path),
        "--config-file",
        str(CONFIG_FILE),
        "--vit-path",
        str(QWEN_SNAPSHOT),
        "--output-dir",
        str(output_dir),
    ]
    print(f"[export] step={step} checkpoint={checkpoint_path} output={output_dir}")
    if args.dry_run:
        print("          command:", " ".join(command))
        return
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    validate_inputs(args.steps, args.exporter, args.output_root)
    env = build_env()
    for step in args.steps:
        export_step(step, args, env)


if __name__ == "__main__":
    main()
