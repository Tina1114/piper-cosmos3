#!/usr/bin/env python3
"""Adversarial readiness audit for Cosmos3-Edge Piper14 SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = ROOT / "external" / "cosmos" / "packages" / "cosmos3"
for path in (ROOT, FRAMEWORK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from piper_cosmos.cosmos3.local_edge_assets import DEFAULT_EDGE_SNAPSHOT, DEFAULT_WAN_VAE_PATH  # noqa: E402
from piper_cosmos.cosmos3.local_edge_assets import resolve_local_edge_assets  # noqa: E402
from scripts.preflight_cosmos3_piper14_sft import PreflightInputs, run_preflight  # noqa: E402
from scripts.verify_cosmos3_dcp import verify_dcp  # noqa: E402

EXPECTED_COSMOS_COMMIT = "7c22d8aa2e97adcd7857399d1ff1b088be1ff401"
EXPECTED_FRAMEWORK_BASE_COMMIT = "f734253f0f6af3e268372402f44435c38f55ef3e"
EXPECTED_EDGE_REVISION = "6f58f6b4c91288838e60b6bcb2cc45d997e961de"
EXPECTED_SKIP_KEYS = {
    "net_ema.",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "action_pos_embed",
}
EXPECTED_TRAINABLE_KEYS = {
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "k_norm_und_for_gen",
    "action2llm",
    "llm2action",
    "action_modality_embed",
}


def check(check_id: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"id": check_id, "status": "passed" if passed else "failed", "details": details}


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate_report(path: Path, toml: Path, framework_commit: str) -> dict[str, Any]:
    if not path.is_file():
        return check("adversarial_gate", False, path=str(path), reason="missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return check("adversarial_gate", False, path=str(path), reason=str(exc))
    required = {
        "status": payload.get("status") == "passed",
        "edge_revision": payload.get("edge_revision") == EXPECTED_EDGE_REVISION,
        "framework_commit": payload.get("framework_commit") == framework_commit,
        "toml_sha256": payload.get("toml_sha256") == sha256(toml),
        "world_size": payload.get("world_size") == 4,
        "steps_completed": int(payload.get("steps_completed", 0)) >= 100,
        "finite_losses": payload.get("finite_losses") is True,
        "action_head_gradient_nonzero": payload.get("action_head_gradient_nonzero") is True,
        "reasoner_gradient_zero": payload.get("reasoner_gradient_zero") is True,
        "checkpoint_reload": payload.get("checkpoint_reload") is True,
        "action_inference_shape": payload.get("action_inference_shape") == [32, 14],
    }
    return check("adversarial_gate", all(required.values()), path=str(path), assertions=required)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        assets = resolve_local_edge_assets(edge_snapshot=args.edge_snapshot, wan_vae_path=args.wan_vae)
        checks.append(check("offline_assets", True, paths={key: str(value) for key, value in assets.items()}))
    except FileNotFoundError as exc:
        checks.append(check("offline_assets", False, reason=str(exc)))

    cosmos_commit = git_output(ROOT / "external" / "cosmos", "rev-parse", "HEAD")
    framework_commit = git_output(FRAMEWORK_ROOT, "rev-parse", "HEAD")
    framework_base = git_output(FRAMEWORK_ROOT, "merge-base", framework_commit, EXPECTED_FRAMEWORK_BASE_COMMIT)
    checks.extend(
        [
            check("cosmos_commit", cosmos_commit == EXPECTED_COSMOS_COMMIT, actual=cosmos_commit),
            check(
                "framework_base",
                framework_base == EXPECTED_FRAMEWORK_BASE_COMMIT,
                actual_commit=framework_commit,
                merge_base=framework_base,
            ),
            check(
                "framework_clean",
                git_output(FRAMEWORK_ROOT, "status", "--porcelain") == "",
                status=git_output(FRAMEWORK_ROOT, "status", "--porcelain"),
            ),
        ]
    )

    nemotron_config = (
        FRAMEWORK_ROOT
        / "cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
    )
    hidden_size = json.loads(nemotron_config.read_text(encoding="utf-8"))["hidden_size"]
    checks.append(check("edge_hidden_size", hidden_size == 2048, hidden_size=hidden_size))

    from cosmos_framework.data.generator.action import domain_utils

    collisions = {
        name: domain_id
        for name, domain_id in domain_utils.EMBODIMENT_TO_DOMAIN_ID.items()
        if domain_id == 21 and name != "piper14"
    }
    checks.append(check("domain_21_collision", not collisions, collisions=collisions))

    from piper_cosmos.cosmos3.action_policy_piper14_edge import action_policy_piper14_edge as config

    skip_keys = set(config["checkpoint"]["keys_to_skip_loading"])
    trainable_keys = set(config["optimizer"]["keys_to_select"])
    model_config = config["model"]["config"]
    checks.extend(
        [
            check("fresh_action_head", skip_keys == EXPECTED_SKIP_KEYS, actual=sorted(skip_keys)),
            check("trainable_allowlist", trainable_keys == EXPECTED_TRAINABLE_KEYS, actual=sorted(trainable_keys)),
            check(
                "model_contract",
                model_config["action_gen"]
                and model_config["vision_gen"]
                and not model_config["sound_gen"]
                and model_config["resolution"] == "480"
                and model_config["tokenizer"]["encode_exact_durations"] == [33]
                and model_config["rectified_flow_training_config"]["action_loss_weight"] == 10.0,
                action_gen=model_config["action_gen"],
                vision_gen=model_config["vision_gen"],
                sound_gen=model_config["sound_gen"],
                resolution=model_config["resolution"],
            ),
        ]
    )

    with args.toml.open("rb") as handle:
        toml = tomllib.load(handle)
    checks.append(
        check(
            "nano20k_control_variables",
            toml["job"]["project"] == "cosmos_battery"
            and toml["job"]["wandb_mode"] == "online"
            and toml["trainer"]["max_iter"] == 20_000
            and toml["trainer"]["grad_accum_iter"] == 1
            and toml["scheduler"]["cycle_lengths"] == [20_000]
            and toml["dataloader_train"]["max_samples_per_batch"] == 8
            and toml["checkpoint"]["save_iter"] == 500
            and toml["model"]["parallelism"]["data_parallel_shard_degree"] == 4,
            job=toml.get("job"),
            trainer=toml.get("trainer"),
            scheduler=toml.get("scheduler"),
        )
    )

    from piper_cosmos.cosmos3.piper14_hdf5_action_dataset import (
        Piper14HDF5ActionDataset,
        get_piper14_hdf5_sft_dataset,
    )

    raw_dataset = Piper14HDF5ActionDataset(root=str(args.data_root), config_path=str(args.data_config))
    raw = raw_dataset[0]
    transformed = get_piper14_hdf5_sft_dataset(
        root=str(args.data_root),
        config_path=str(args.data_config),
        tokenizer_config=None,
        format_prompt_as_json=True,
    )[0]
    plan = transformed["sequence_plan"].as_dict()
    checks.append(
        check(
            "real_dataset_contract",
            len(raw_dataset.episode_paths) == 138
            and tuple(raw["video"].shape) == (3, 33, 720, 640)
            and tuple(raw["action"].shape) == (33, 14)
            and tuple(transformed["video"].shape) == (3, 33, 640, 640)
            and tuple(transformed["action"].shape) == (33, 64)
            and int(transformed["raw_action_dim"]) == 14
            and plan["condition_frame_indexes_vision"] == [0]
            and plan["condition_frame_indexes_action"] == [0],
            episodes=len(raw_dataset.episode_paths),
            windows=len(raw_dataset),
            raw_video_shape=list(raw["video"].shape),
            raw_action_shape=list(raw["action"].shape),
            transformed_video_shape=list(transformed["video"].shape),
            transformed_action_shape=list(transformed["action"].shape),
            sequence_plan=plan,
        )
    )

    preflight = run_preflight(
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
    checks.append(check("generic_preflight", preflight["status"] == "ready", report=preflight))
    dcp = verify_dcp(args.base_checkpoint, require_checkpoint_json=False)
    checks.append(check("edge_dcp", dcp["status"] == "passed", report=dcp))
    if dcp["status"] == "passed":
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(str(args.base_checkpoint / "model")).read_metadata()
        wanted = {
            "net.action_modality_embed": (2048,),
            "net.action2llm.fc.weight": (32, 64 * 2048),
            "net.action2llm.bias.weight": (32, 2048),
            "net.llm2action.fc.weight": (32, 2048 * 64),
            "net.llm2action.bias.weight": (32, 64),
        }
        actual = {
            key: tuple(metadata.state_dict_metadata[key].size)
            for key in wanted
            if key in metadata.state_dict_metadata
        }
        checks.append(check("edge_dcp_action_shapes", actual == wanted, expected=wanted, actual=actual))

    if args.require_adversarial_gate:
        checks.append(validate_gate_report(args.adversarial_gate, args.toml, framework_commit))

    failed = [item["id"] for item in checks if item["status"] == "failed"]
    return {
        "status": "ready" if not failed else "not_ready",
        "ready_for_sft": not failed,
        "failed_checks": failed,
        "checks": checks,
        "audit": {
            "edge_revision": EXPECTED_EDGE_REVISION,
            "cosmos_commit": cosmos_commit,
            "framework_commit": framework_commit,
            "toml_sha256": sha256(args.toml),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3/.venv/bin/python"),
    )
    parser.add_argument("--toml", type=Path, default=ROOT / "configs/cosmos3/sft/action_policy_piper14_edge.toml")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/project/peilab/wam/physical_WM/data/battery_assemble/perfect"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=ROOT / "configs/dataset_configs/battery_assemble_hdf5.yaml",
    )
    parser.add_argument(
        "--edge-snapshot",
        type=Path,
        default=Path(os.environ.get("COSMOS3_EDGE_SNAPSHOT", str(DEFAULT_EDGE_SNAPSHOT))),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "BASE_CHECKPOINT_PATH",
                str(ROOT / "external/cosmos/checkpoints/Cosmos3-Edge-DCP"),
            )
        ),
    )
    parser.add_argument(
        "--wan-vae",
        type=Path,
        default=Path(os.environ.get("WAN_VAE_PATH", str(DEFAULT_WAN_VAE_PATH))),
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/cosmos3_edge_piper14")
    parser.add_argument("--require-slurm-account", action="store_true")
    parser.add_argument("--require-adversarial-gate", action="store_true")
    parser.add_argument(
        "--adversarial-gate",
        type=Path,
        default=ROOT / "reports/cosmos3_edge_piper14/adversarial_gate.json",
    )
    parser.add_argument("--report", type=Path, default=ROOT / "reports/cosmos3_edge_piper14/readiness.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
