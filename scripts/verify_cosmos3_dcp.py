#!/usr/bin/env python3
"""Verify a Cosmos3 DCP checkpoint directory before using it for SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify_dcp(path: Path) -> dict[str, Any]:
    model_dir = path / "model"
    metadata = model_dir / ".metadata"
    checkpoint_json = path / "checkpoint.json"
    model_config = model_dir / "config.json"
    distcp_files = sorted(model_dir.glob("*.distcp")) if model_dir.is_dir() else []

    missing: list[str] = []
    if not path.is_dir():
        missing.append("dcp_dir")
    if not model_dir.is_dir():
        missing.append("model_dir")
    if not metadata.is_file():
        missing.append("model_metadata")
    if not distcp_files:
        missing.append("distcp_shards")
    if not checkpoint_json.is_file():
        missing.append("checkpoint_json")
    if not model_config.is_file():
        missing.append("model_config")

    return {
        "status": "passed" if not missing else "failed",
        "path": str(path),
        "model_dir": str(model_dir),
        "has_metadata": metadata.is_file(),
        "distcp_files": len(distcp_files),
        "checkpoint_json": str(checkpoint_json),
        "has_checkpoint_json": checkpoint_json.is_file(),
        "model_config": str(model_config),
        "has_model_config": model_config.is_file(),
        "missing": missing,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("external/cosmos/checkpoints/Cosmos3-Nano-DCP"))
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_dcp(args.path)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
