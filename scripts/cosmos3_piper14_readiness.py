#!/usr/bin/env python3
"""Summarize whether Cosmos3 Piper14 SFT can be submitted."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_cosmos3_piper14_sft import DEFAULT_BASE_CHECKPOINT
from scripts.preflight_cosmos3_piper14_sft import PreflightInputs, run_preflight
from scripts.verify_cosmos3_dcp import verify_dcp
from piper_cosmos.cosmos3.local_hf_assets import DEFAULT_WAN_VAE_PATH
from piper_cosmos.cosmos3.local_hf_assets import ENV_WAN_VAE_PATH


@dataclass(frozen=True)
class ReadinessInputs:
    python_bin: Path
    toml: Path
    data_root: Path
    data_config: Path
    base_checkpoint: Path
    wan_vae: Path
    output_root: Path
    require_slurm_account: bool
    env: Mapping[str, str]


def build_readiness_report(inputs: ReadinessInputs) -> dict[str, Any]:
    dcp_report = verify_dcp(inputs.base_checkpoint)
    sft_preflight = run_preflight(
        PreflightInputs(
            python_bin=inputs.python_bin,
            toml=inputs.toml,
            data_root=inputs.data_root,
            data_config=inputs.data_config,
            base_checkpoint=inputs.base_checkpoint,
            wan_vae=inputs.wan_vae,
            output_root=inputs.output_root,
            require_slurm_account=inputs.require_slurm_account,
            env=inputs.env,
        )
    )
    failed_checks: list[str] = []
    if dcp_report["status"] != "passed":
        failed_checks.append("dcp")
    failed_checks.extend(f"sft_preflight:{item}" for item in sft_preflight.get("failed_checks", []))
    ready = not failed_checks
    return {
        "status": "ready" if ready else "not_ready",
        "ready_for_sft": ready,
        "failed_checks": failed_checks,
        "dcp": dcp_report,
        "sft_preflight": sft_preflight,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=Path, default=Path("external/cosmos/packages/cosmos3/.venv/bin/python"))
    parser.add_argument("--toml", type=Path, default=Path("configs/cosmos3/sft/action_policy_piper14_nano.toml"))
    parser.add_argument("--data-root", type=Path, default=Path("/project/peilab/wam/physical_WM/data/battery_assemble/perfect"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/dataset_configs/battery_assemble_hdf5.yaml"))
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(os.environ.get("BASE_CHECKPOINT_PATH", str(DEFAULT_BASE_CHECKPOINT))),
    )
    parser.add_argument(
        "--wan-vae",
        type=Path,
        default=Path(os.environ.get("WAN_VAE_PATH") or os.environ.get(ENV_WAN_VAE_PATH, str(DEFAULT_WAN_VAE_PATH))),
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports/cosmos3_piper14"))
    parser.add_argument("--require-slurm-account", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("reports/cosmos3_piper14/readiness.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_readiness_report(
        ReadinessInputs(
            python_bin=args.python_bin,
            toml=args.toml,
            data_root=args.data_root,
            data_config=args.data_config,
            base_checkpoint=args.base_checkpoint,
            wan_vae=args.wan_vae,
            output_root=args.output_root,
            require_slurm_account=args.require_slurm_account,
            env=os.environ,
        )
    )
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
