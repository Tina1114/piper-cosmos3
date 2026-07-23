#!/usr/bin/env python3
"""Combine completed smoke and resume audits into the fail-closed training gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EDGE_REVISION = "6f58f6b4c91288838e60b6bcb2cc45d997e961de"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def slurm_state(job_id: str) -> str:
    output = subprocess.check_output(
        ["sacct", "-X", "-j", job_id, "--noheader", "--format=State"],
        text=True,
    )
    states = [line.strip().split()[0].split("+")[0] for line in output.splitlines() if line.strip()]
    if not states:
        raise RuntimeError(f"sacct returned no state for job {job_id}")
    return states[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, required=True)
    parser.add_argument("--smoke-job", required=True)
    parser.add_argument("--reload-job", required=True)
    parser.add_argument(
        "--toml",
        type=Path,
        default=ROOT / "configs/cosmos3/sft/action_policy_piper14_edge.toml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/cosmos3_edge_piper14/adversarial_gate.json",
    )
    args = parser.parse_args()

    smoke = load_json(args.smoke_report)
    reload_report = load_json(args.reload_report)
    smoke_state = slurm_state(args.smoke_job)
    reload_state = slurm_state(args.reload_job)
    inference = reload_report.get("action_inference_result") or {}
    checks = {
        "smoke_slurm_completed": smoke_state == "COMPLETED",
        "reload_slurm_completed": reload_state == "COMPLETED",
        "smoke_runtime_passed": smoke.get("status") == "passed",
        "reload_runtime_passed": reload_report.get("status") == "passed",
        "world_size_4": smoke.get("world_size") == 4 and reload_report.get("world_size") == 4,
        "steps_100": int(smoke.get("iteration_end", 0)) >= 100,
        "losses_finite": smoke.get("finite_losses") is True,
        "modality_losses_finite": smoke.get("finite_modality_losses") is True,
        "modality_losses_nonzero": smoke.get("nonzero_modality_losses") is True,
        "action_gradient": smoke.get("action_head_gradient_nonzero") is True,
        "action_gradient_finite": smoke.get("action_head_gradient_finite") is True,
        "frozen_gradient_zero": smoke.get("frozen_gradient_zero") is True,
        "checkpoint_reload": (
            reload_report.get("checkpoint_reload") is True
            and int(reload_report.get("load_iteration", 0)) >= 100
        ),
        "action_shape": inference.get("shape") == [32, 14],
        "action_finite": inference.get("finite") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    framework_root = ROOT / "external/cosmos/packages/cosmos3"
    payload = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checks": checks,
        "edge_revision": EXPECTED_EDGE_REVISION,
        "framework_commit": git_output(framework_root, "rev-parse", "HEAD"),
        "outer_commit": git_output(ROOT, "rev-parse", "HEAD"),
        "toml_sha256": sha256(args.toml),
        "world_size": smoke.get("world_size"),
        "steps_completed": smoke.get("iteration_end"),
        "finite_losses": smoke.get("finite_losses"),
        "action_head_gradient_nonzero": smoke.get("action_head_gradient_nonzero"),
        "reasoner_gradient_zero": smoke.get("frozen_gradient_zero"),
        "checkpoint_reload": checks["checkpoint_reload"],
        "action_inference_shape": inference.get("shape"),
        "action_inference_finite": inference.get("finite"),
        "smoke_job": args.smoke_job,
        "reload_job": args.reload_job,
        "smoke_slurm_state": smoke_state,
        "reload_slurm_state": reload_state,
        "smoke_report": str(args.smoke_report.resolve()),
        "reload_report": str(args.reload_report.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)
    if failures:
        raise SystemExit(f"smoke gate failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
