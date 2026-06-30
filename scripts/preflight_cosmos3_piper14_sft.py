#!/usr/bin/env python3
"""Preflight checks for the real Cosmos3 Piper14 SFT SLURM job."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import tomllib
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping

from piper_cosmos.cosmos3.local_hf_assets import DEFAULT_WAN_VAE_PATH
from piper_cosmos.cosmos3.local_hf_assets import ENV_WAN_VAE_PATH


WAN_VAE_CANDIDATE_HINTS_ENV = "WAN_VAE_CANDIDATE_HINTS"
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CHECKPOINT = SCRIPT_ROOT / "external/cosmos/checkpoints/Cosmos3-Nano-DCP"
DEFAULT_WAN_VAE_CANDIDATE_HINTS = (DEFAULT_WAN_VAE_PATH,)


@dataclass(frozen=True)
class PreflightInputs:
    python_bin: Path
    toml: Path
    data_root: Path
    data_config: Path
    base_checkpoint: Path
    wan_vae: Path
    output_root: Path
    require_slurm_account: bool
    env: Mapping[str, str]
    search_roots: tuple[Path, ...] = field(default_factory=tuple)


def check(check_id: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": check_id, "status": "passed" if passed else "failed", "details": details}


def skipped(check_id: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": check_id, "status": "skipped", "details": details}


def check_python_bin(path: Path) -> dict[str, Any]:
    path_s = str(path)
    exists = path.exists()
    executable = os.access(path, os.X_OK)
    fastwam = "fastwam" in path_s.lower()
    return check(
        "python_bin",
        exists and executable and not fastwam,
        {
            "path": path_s,
            "exists": exists,
            "executable": executable,
            "fastwam_path": fastwam,
            "expected_default": "external/cosmos/packages/cosmos3/.venv/bin/python",
        },
    )


def _candidate_wan_vae(search_roots: tuple[Path, ...], env: Mapping[str, str] | None = None) -> Path | None:
    # Keep WAN VAE resolution deterministic: only explicit candidate hints or the
    # repo-maintained default candidate are used. No filesystem crawling.
    env = env or os.environ
    raw_hint = env.get(WAN_VAE_CANDIDATE_HINTS_ENV, "").strip()
    candidates: list[Path] = []
    if raw_hint:
        for raw in raw_hint.split(os.pathsep):
            path = Path(raw.strip())
            if str(path):
                candidates.append(path)
    else:
        candidates.extend(DEFAULT_WAN_VAE_CANDIDATE_HINTS)
    if not candidates:
        return None
    candidates = list(dict.fromkeys(candidates))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _candidate_hf_checkpoint(search_roots: tuple[Path, ...]) -> Path | None:
    for root in search_roots:
        for path in sorted(root.rglob("Cosmos3-Nano")):
            if path.is_dir() and (path / "model_index.json").is_file():
                return path
    return None


def check_file(
    check_id: str,
    path: Path,
    search_roots: tuple[Path, ...] = (),
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    provided = str(path) not in {"", "."}
    details = {"path": str(path), "provided": provided, "exists": path.exists(), "is_file": path.is_file()}
    if check_id == "wan_vae" and not (provided and path.exists() and path.is_file()):
        candidate = _candidate_wan_vae(search_roots, env=env)
        if candidate is not None:
            details["candidate_wan_vae"] = str(candidate)
    return check(
        check_id,
        provided and path.exists() and path.is_file(),
        details,
    )


def resolve_wan_vae_path(
    provided: Path,
    search_roots: tuple[Path, ...],
    env: Mapping[str, str] | None = None,
) -> tuple[Path, bool]:
    """Resolve WAN_VAE path with fallback discovery if caller did not pass a valid file."""

    if provided.exists() and provided.is_file():
        return provided, False
    candidate = _candidate_wan_vae(search_roots, env=env)
    if candidate is not None:
        return candidate, True
    return provided, False


def check_directory(check_id: str, path: Path, search_roots: tuple[Path, ...] = ()) -> dict[str, Any]:
    provided = str(path) not in {"", "."}
    passed = provided and path.exists() and path.is_dir()
    details = {"path": str(path), "provided": provided, "exists": path.exists(), "is_dir": path.is_dir()}
    if check_id == "base_checkpoint" and passed:
        model_dir = path / "model"
        metadata = model_dir / ".metadata"
        distcp_files = list(model_dir.glob("*.distcp")) if model_dir.is_dir() else []
        details.update(
            {
                "model_dir": str(model_dir),
                "has_model_dir": model_dir.is_dir(),
                "has_metadata": metadata.is_file(),
                "distcp_files": len(distcp_files),
            }
        )
        passed = model_dir.is_dir() and metadata.is_file() and len(distcp_files) > 0
    if check_id == "base_checkpoint" and not (provided and path.exists() and path.is_dir()):
        candidate = _candidate_hf_checkpoint(search_roots)
        if candidate is not None:
            suggested_output = candidate.parent / "Cosmos3-Nano-DCP"
            details.update(
                {
                    "candidate_hf_checkpoint": str(candidate),
                    "suggested_dcp_output": str(suggested_output),
                    "suggested_convert_command": (
                        "cd external/cosmos/packages/cosmos3 && "
                        "PYTHONPATH=. python -m cosmos_framework.scripts.convert_model_to_dcp "
                        f"-o {suggested_output.resolve()} --checkpoint-path {candidate.resolve()}"
                    ),
                    "suggested_slurm_command": "BATTERY_SLURM_ACCOUNT=<account> bash scripts/submit_convert_cosmos3_nano_dcp.sh",
                }
            )
    return check(
        check_id,
        passed,
        details,
    )


def check_battery_data(data_root: Path) -> dict[str, Any]:
    files = []
    if data_root.exists():
        files = sorted(data_root.rglob("*.hdf5")) + sorted(data_root.rglob("*.h5"))
    return check(
        "battery_data",
        data_root.exists() and data_root.is_dir() and len(set(files)) > 0,
        {
            "data_root": str(data_root),
            "exists": data_root.exists(),
            "is_dir": data_root.is_dir(),
            "hdf5_files": len(set(files)),
        },
    )


def check_output_root(path: Path) -> dict[str, Any]:
    exists = path.exists()
    nearest_parent = path.parent
    while not nearest_parent.exists() and nearest_parent != nearest_parent.parent:
        nearest_parent = nearest_parent.parent
    writable_parent = nearest_parent.exists() and nearest_parent.is_dir() and os.access(nearest_parent, os.W_OK)
    writable_existing = exists and path.is_dir() and os.access(path, os.W_OK)
    return check(
        "output_root",
        (writable_existing if exists else writable_parent),
        {
            "path": str(path),
            "exists": exists,
            "is_dir": path.is_dir(),
            "will_create": not exists,
            "nearest_existing_parent": str(nearest_parent),
            "parent_writable": writable_parent,
            "writable": writable_existing if exists else writable_parent,
        },
    )


def check_slurm_account(env: Mapping[str, str], required: bool) -> dict[str, Any]:
    account = env.get("BATTERY_SLURM_ACCOUNT", "")
    sbatch_path = shutil.which("sbatch")
    details = {
        "required": required,
        "BATTERY_SLURM_ACCOUNT_set": bool(account),
        "sbatch": sbatch_path,
    }
    if not required:
        return skipped("slurm_account", details)
    return check("slurm_account", bool(account) and bool(sbatch_path), details)


def _job_wandb_mode_from_toml(path: Path) -> tuple[str | None, str]:
    if not path.exists() or not path.is_file():
        return None, "toml_missing"
    with path.open("rb") as f:
        payload = tomllib.load(f)
    job = payload.get("job")
    if not isinstance(job, dict):
        return None, "toml_missing_job_section"
    mode = job.get("wandb_mode")
    if not isinstance(mode, str) or not mode.strip():
        return None, "toml_missing_job.wandb_mode"
    return mode.strip(), "toml"


def resolve_wandb_mode(toml: Path, env: Mapping[str, str]) -> tuple[str | None, str]:
    mode, source = _job_wandb_mode_from_toml(toml)
    overrides = env.get("EXTRA_TAIL_OVERRIDES", "")
    if not overrides.strip():
        return mode, source
    for token in shlex.split(overrides):
        if token.startswith("job.wandb_mode="):
            return token.split("=", 1)[1].strip(), "EXTRA_TAIL_OVERRIDES"
    return mode, source


def check_wandb_auth(toml: Path, env: Mapping[str, str]) -> dict[str, Any]:
    mode, mode_source = resolve_wandb_mode(toml, env)
    batch_home = Path(env.get("HOME_DIR") or env.get("HOME") or Path.home())
    netrc_path = batch_home / ".netrc"
    api_key = env.get("WANDB_API_KEY", "").strip()
    api_key_file_raw = env.get("WANDB_API_KEY_FILE", "").strip()
    api_key_file = Path(api_key_file_raw).expanduser() if api_key_file_raw else None
    api_key_file_exists = bool(api_key_file and api_key_file.is_file())
    api_key_file_nonempty = bool(api_key_file_exists and api_key_file.stat().st_size > 0)
    details = {
        "wandb_mode": mode,
        "wandb_mode_source": mode_source,
        "batch_home": str(batch_home),
        "WANDB_API_KEY_set": bool(api_key),
        "WANDB_API_KEY_FILE": str(api_key_file) if api_key_file is not None else "",
        "WANDB_API_KEY_FILE_exists": api_key_file_exists,
        "WANDB_API_KEY_FILE_nonempty": api_key_file_nonempty,
        "netrc_path": str(netrc_path),
        "netrc_exists": netrc_path.is_file(),
    }
    if mode != "online":
        return skipped("wandb_auth", details)
    passed = bool(api_key) or api_key_file_nonempty or netrc_path.is_file()
    if not passed:
        details["failure_reason"] = (
            "wandb_mode=online requires batch-visible credentials: set WANDB_API_KEY, "
            "set WANDB_API_KEY_FILE to a non-empty file, or ensure batch HOME/HOME_DIR has .netrc."
        )
    return check("wandb_auth", passed, details)


def run_preflight(inputs: PreflightInputs) -> dict[str, Any]:
    search_roots = inputs.search_roots or (Path("external/cosmos/checkpoints"),)
    resolved_wan_vae, used_candidate = resolve_wan_vae_path(inputs.wan_vae, search_roots, env=inputs.env)
    wan_vae_details = {"wan_vae_resolved_from_candidate": used_candidate}
    if used_candidate:
        wan_vae_details["wan_vae_resolved_path"] = str(resolved_wan_vae)
        wan_vae_details["candidate_wan_vae"] = str(resolved_wan_vae)
    wan_vae_check = check_file("wan_vae", resolved_wan_vae, search_roots, env=inputs.env)
    if isinstance(wan_vae_check.get("details"), dict):
        wan_vae_check["details"] = {**wan_vae_check["details"], **wan_vae_details}
    checks = [
        check_python_bin(inputs.python_bin),
        check_file("toml", inputs.toml),
        check_battery_data(inputs.data_root),
        check_file("data_config", inputs.data_config),
        check_directory("base_checkpoint", inputs.base_checkpoint, search_roots),
        wan_vae_check,
        check_output_root(inputs.output_root),
        check_slurm_account(inputs.env, inputs.require_slurm_account),
        check_wandb_auth(inputs.toml, inputs.env),
    ]
    failed = [item for item in checks if item["status"] == "failed"]
    return {
        "status": "ready" if not failed else "not_ready",
        "checks": checks,
        "failed_checks": [item["id"] for item in failed],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cosmos3 Piper14 SFT preflight checks.")
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
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight(
        PreflightInputs(
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
    text = json.dumps(report, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
