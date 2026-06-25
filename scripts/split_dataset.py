#!/usr/bin/env python3
"""Create deterministic episode-level train/val/test splits for HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.data.hdf5_reader import find_hdf5_files


DEFAULT_DATA_ROOT = Path("/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect")
DEFAULT_OUTPUT = Path("reports/pack_3_objects_plus_split.json")


def split_paths(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio /= total_ratio
    val_ratio /= total_ratio

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(round(n * train_ratio))
    val_end = train_end + int(round(n * val_ratio))
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return {
        "train": [str(path) for path in train],
        "val": [str(path) for path in val],
        "test": [str(path) for path in test],
    }


def build_split(args: argparse.Namespace) -> dict[str, Any]:
    episodes = find_hdf5_files(args.data_root)
    if not episodes:
        raise SystemExit(f"No HDF5 episodes found under {args.data_root}")
    splits = split_paths(
        episodes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return {
        "data_root": str(args.data_root),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "num_episodes": len(episodes),
        "splits": splits,
        "counts": {name: len(paths) for name, paths in splits.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write deterministic episode-level train/val/test splits for Piper HDF5 data."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="HDF5 episode root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = build_split(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
        f.write("\n")
    print(json.dumps(split["counts"], indent=2))


if __name__ == "__main__":
    main()
