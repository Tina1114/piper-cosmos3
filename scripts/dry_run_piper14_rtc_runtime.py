#!/usr/bin/env python3
"""Run a no-robot RTC loop against a Cosmos Piper14 policy server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.deployment.cosmos_piper14_remote_client import CosmosPiper14RemotePolicyClient  # noqa: E402
from piper_cosmos.deployment.piper14_rtc_runtime import (  # noqa: E402
    HDF5ObservationSource,
    Piper14RTCRuntime,
    Piper14RTCRuntimeConfig,
    RecordingActionSink,
    write_json_report,
)


DEFAULT_DATA_ROOT = Path("/project/peilab/wam/physical_WM/data/battery_assemble/perfect")
DEFAULT_DATA_CONFIG = ROOT / "configs/dataset_configs/battery_assemble_hdf5.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--authkey", default="cosmos-piper14")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--episode", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--exp-weight-factor", type=float, default=0.5)
    parser.add_argument("--sleep", action="store_true", help="Sleep to match --control-hz during dry-run.")
    parser.add_argument("--loop", action="store_true", help="Loop the HDF5 episode if --steps exceeds episode length.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--timing", action="store_true", help="Print RPC send/receive timings.")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = HDF5ObservationSource.from_data_root(
        data_root=args.data_root,
        config_path=args.data_config,
        episode=args.episode,
        prompt=args.prompt,
        loop=args.loop,
    )
    sink = RecordingActionSink()
    config = Piper14RTCRuntimeConfig(
        max_steps=args.steps,
        control_hz=args.control_hz,
        chunk_size=args.chunk_size,
        replan_interval=args.replan_interval,
        exp_weight_factor=args.exp_weight_factor,
        sleep=args.sleep,
        prompt=args.prompt,
        debug=args.debug,
    )

    with CosmosPiper14RemotePolicyClient(
        host=args.host,
        port=args.port,
        authkey=args.authkey,
        timing=args.timing,
    ) as policy:
        metadata = policy.metadata()
        runtime = Piper14RTCRuntime(policy=policy, observation_source=source, action_sink=sink, config=config)
        report = runtime.run()

    report = {
        "metadata": metadata,
        "episode": str(source.episode_path),
        "data_config": str(source.config_path),
        **report,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        write_json_report(args.report, report)
    if report["starved_steps"]:
        raise SystemExit(f"RTC dry-run starved for {report['starved_steps']} step(s).")


if __name__ == "__main__":
    main()
