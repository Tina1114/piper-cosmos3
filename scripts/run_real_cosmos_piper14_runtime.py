#!/usr/bin/env python3
"""Run real Piper14 RTC runtime against a Cosmos Piper14 policy server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.deployment.real_robot_runtime import (  # noqa: E402
    load_mapping_config,
    run_real_cosmos_piper14_runtime,
)


DEFAULT_CONFIG = Path("/home/agilex/World_Action_Model/physical_WM/configs/real_deploy_rtc_fastwam.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=None, help="Override Cosmos policy server host.")
    parser.add_argument("--port", type=int, default=8766, help="Cosmos policy server port.")
    parser.add_argument("--authkey", default="cosmos-piper14")
    parser.add_argument("--execute-actions", dest="execute_actions", action="store_true", default=None)
    parser.add_argument("--no-execute-actions", dest="execute_actions", action="store_false")
    parser.add_argument("--move-to-initial", dest="move_to_initial", action="store_true", default=None)
    parser.add_argument("--no-move-to-initial", dest="move_to_initial", action="store_false")
    parser.add_argument("--no-robot", action="store_true", help="Skip CAN/pyAgxArm and only test cameras + policy.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--replan-interval", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    config.setdefault("policy_server", {})
    config.setdefault("runtime", {})
    config.setdefault("fastwam", {})
    if args.host is not None:
        config["policy_server"]["host"] = args.host
    config["policy_server"]["port"] = args.port
    config["policy_server"]["authkey"] = args.authkey
    if args.execute_actions is not None:
        config["runtime"]["execute_actions"] = args.execute_actions
    if args.move_to_initial is not None:
        config["runtime"]["move_to_initial"] = args.move_to_initial
    if args.no_robot:
        config["runtime"]["no_robot"] = True
    if args.max_steps is not None:
        config["runtime"]["max_steps"] = args.max_steps
    if args.control_hz is not None:
        config["runtime"]["rospy_rate"] = args.control_hz
    if args.replan_interval is not None:
        config["runtime"]["replan_interval"] = args.replan_interval
    if args.output_dir is not None:
        config["runtime"]["output_dir"] = args.output_dir
    if args.prompt is not None:
        config["fastwam"]["prompt"] = args.prompt
    # Cosmos Piper14 policy is 14-dim only in this repo.
    config["runtime"]["arm_mode"] = "dual"
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_mapping_config(args.config), args)
    report = run_real_cosmos_piper14_runtime(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
