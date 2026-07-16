#!/usr/bin/env python3
"""Start the RTC-style Cosmos Piper14 policy server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.deployment.cosmos_piper14_policy import CosmosPiper14PolicyConfig
from piper_cosmos.deployment.cosmos_piper14_policy_server import serve_cosmos_piper14_policy
from piper_cosmos.cosmos3.local_hf_assets import bootstrap_local_hf_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "cosmos_battery" / "20k"))
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--authkey", default="cosmos-piper14")
    parser.add_argument("--prompt", default="Assemble the mouse's battery.")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--resolution", default="480")
    parser.add_argument("--mock-backend", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CosmosPiper14PolicyConfig(
        checkpoint=args.checkpoint,
        config_file=args.config_file,
        prompt=args.prompt,
        action_horizon=args.action_horizon,
        max_action_dim=args.max_action_dim,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        fps=args.fps,
        seed=args.seed,
        host=args.host,
        port=args.port,
        mock_backend=args.mock_backend,
    )
    bootstrap_local_hf_assets()
    serve_cosmos_piper14_policy(config, host=args.host, port=args.port, authkey=args.authkey)


if __name__ == "__main__":
    main()
