from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_m3_train_and_eval_help() -> None:
    train = run_command(["training/train_baseline_piper14.py", "--help"])
    assert train.returncode == 0, train.stderr
    assert "--max-steps" in train.stdout

    eval_cmd = run_command(["piper_cosmos/eval/offline_action_eval.py", "--help"])
    assert eval_cmd.returncode == 0, eval_cmd.stderr
    assert "--checkpoint" in eval_cmd.stdout


def test_train_smoke_writes_skipped_report_when_torch_missing(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    report = tmp_path / "smoke.json"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"data_root": str(tmp_path), "config_path": "configs/data/piper_dual_hdf5.yaml"},
                "training": {"max_steps": 1, "batch_size": 1, "num_workers": 0},
                "model": {"action_horizon": 16, "action_dim": 14},
            }
        ),
        encoding="utf-8",
    )

    result = run_command(
        [
            "training/train_baseline_piper14.py",
            "--config",
            str(config),
            "--max-steps",
            "1",
            "--smoke-report",
            str(report),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] in {"passed", "skipped"}
    if payload["status"] == "skipped":
        assert payload["reason"] == "torch_not_available"
