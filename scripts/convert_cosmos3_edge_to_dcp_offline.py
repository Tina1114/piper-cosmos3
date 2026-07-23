#!/usr/bin/env python3
"""Convert a fixed local Cosmos3-Edge HF snapshot to DCP without network access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COSMOS3_ROOT = REPO_ROOT / "external" / "cosmos" / "packages" / "cosmos3"
for path in (REPO_ROOT, COSMOS3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from piper_cosmos.cosmos3.local_edge_assets import DEFAULT_EDGE_SNAPSHOT  # noqa: E402
from piper_cosmos.cosmos3.local_edge_assets import DEFAULT_WAN_VAE_PATH  # noqa: E402
from piper_cosmos.cosmos3.local_edge_assets import bootstrap_local_edge_assets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_EDGE_SNAPSHOT)
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=REPO_ROOT / "external" / "cosmos" / "checkpoints" / "Cosmos3-Edge-DCP",
    )
    parser.add_argument("--wan-vae-path", type=Path, default=DEFAULT_WAN_VAE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_local_edge_assets(edge_snapshot=args.checkpoint_path, wan_vae_path=args.wan_vae_path)
    if args.output_path.exists() and any(args.output_path.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty DCP output: {args.output_path}")

    from cosmos_framework.inference.common.args import CheckpointOverrides
    from cosmos_framework.scripts.convert_model_to_dcp import Args, convert_model_to_dcp

    convert_model_to_dcp(
        Args(
            # Use the registered model name so the converter consumes the
            # framework's complete Cosmos3-Edge YAML. Passing the local HF path
            # directly makes CheckpointOverrides treat root config.json as a
            # training config; that processor-only file has no `ema` block.
            # bootstrap_local_edge_assets() redirects this registry entry back
            # to args.checkpoint_path without network access.
            checkpoint=CheckpointOverrides(checkpoint_path="Cosmos3-Edge"),
            output_path=args.output_path,
        )
    )


if __name__ == "__main__":
    main()
