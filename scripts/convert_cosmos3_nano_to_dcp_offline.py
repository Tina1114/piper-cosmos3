#!/usr/bin/env python3
"""Convert local Cosmos3-Nano HF weights to DCP without network downloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COSMOS3_ROOT = REPO_ROOT / "external" / "cosmos" / "packages" / "cosmos3"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS3_ROOT))
from piper_cosmos.cosmos3.local_hf_assets import DEFAULT_QWEN_SNAPSHOT
from piper_cosmos.cosmos3.local_hf_assets import DEFAULT_WAN_VAE_PATH
from piper_cosmos.cosmos3.local_hf_assets import QWEN_REPOSITORY
from piper_cosmos.cosmos3.local_hf_assets import WAN_VAE_REPOSITORY
from piper_cosmos.cosmos3.local_hf_assets import bootstrap_local_hf_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "external" / "cosmos" / "checkpoints" / "Cosmos3-Nano",
        help="Local Cosmos3-Nano Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=REPO_ROOT / "external" / "cosmos" / "checkpoints" / "Cosmos3-Nano-DCP",
        help="Output DCP checkpoint directory.",
    )
    parser.add_argument(
        "--qwen-snapshot",
        type=Path,
        default=DEFAULT_QWEN_SNAPSHOT,
        help="Local Qwen/Qwen3-VL-8B-Instruct snapshot used for tokenizer files.",
    )
    parser.add_argument(
        "--wan-vae-path",
        type=Path,
        default=DEFAULT_WAN_VAE_PATH,
        help="Local Wan2.2_VAE.pth file used to keep DCP conversion offline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path.is_dir():
        raise SystemExit(f"Missing Cosmos3-Nano checkpoint directory: {args.checkpoint_path}")
    repository_paths = bootstrap_local_hf_assets(
        qwen_snapshot=args.qwen_snapshot,
        wan_vae_path=args.wan_vae_path,
    )

    seeded = {
        QWEN_REPOSITORY: 1 if repository_paths[QWEN_REPOSITORY].is_dir() else 0,
        WAN_VAE_REPOSITORY: 1 if repository_paths[WAN_VAE_REPOSITORY].is_file() else 0,
    }
    if seeded[QWEN_REPOSITORY] == 0:
        raise SystemExit(f"No {QWEN_REPOSITORY} registry entries were seeded.")
    if seeded[WAN_VAE_REPOSITORY] == 0:
        raise SystemExit(f"No {WAN_VAE_REPOSITORY} registry entries were seeded.")

    from cosmos_framework.inference.common.args import CheckpointOverrides
    from cosmos_framework.scripts.convert_model_to_dcp import Args, convert_model_to_dcp

    convert_model_to_dcp(
        Args(
            checkpoint=CheckpointOverrides(checkpoint_path=str(args.checkpoint_path)),
            output_path=args.output_path,
        )
    )


if __name__ == "__main__":
    main()
