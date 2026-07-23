from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = ROOT / "external" / "cosmos" / "packages" / "cosmos3"
for path in (ROOT, FRAMEWORK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from piper_cosmos.cosmos3.action_policy_piper14_edge import action_policy_piper14_edge  # noqa: E402
from piper_cosmos.cosmos3.domain import PIPER14_DOMAIN_ID, register_piper14_domain  # noqa: E402
from piper_cosmos.cosmos3.local_edge_assets import EDGE_REPOSITORY, REQUIRED_EDGE_FILES  # noqa: E402
from piper_cosmos.cosmos3.local_edge_assets import (  # noqa: E402
    WAN_VAE_REPOSITORY,
    resolve_local_edge_assets,
    seed_registry_paths,
)
from scripts.verify_cosmos3_dcp import verify_dcp  # noqa: E402


def test_edge_experiment_matches_audited_baseline() -> None:
    model = action_policy_piper14_edge["model"]["config"]
    assert action_policy_piper14_edge["job"]["project"] == "cosmos_battery"
    assert action_policy_piper14_edge["job"]["wandb_mode"] == "online"
    assert model["vlm_config"]["model_name"] == "nvidia/Cosmos3-Edge-Reasoner"
    assert model["resolution"] == "480"
    assert model["tokenizer"]["encode_exact_durations"] == [33]
    assert model["rectified_flow_training_config"]["action_loss_weight"] == 10.0
    assert "k_norm_und_for_gen" in action_policy_piper14_edge["optimizer"]["keys_to_select"]
    assert set(action_policy_piper14_edge["checkpoint"]["keys_to_skip_loading"]) == {
        "net_ema.",
        "action2llm",
        "llm2action",
        "action_modality_embed",
        "action_pos_embed",
    }


def test_domain_registration_rejects_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.data.generator.action import domain_utils

    monkeypatch.setitem(domain_utils.EMBODIMENT_TO_DOMAIN_ID, "future_official_domain", PIPER14_DOMAIN_ID)
    with pytest.raises(RuntimeError, match="already assigned"):
        register_piper14_domain()


def test_local_edge_assets_require_complete_snapshot(tmp_path: Path) -> None:
    edge = tmp_path / "Cosmos3-Edge"
    for relative in REQUIRED_EDGE_FILES:
        path = edge / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    shard = edge / "transformer" / "dummy.safetensors"
    shard.write_bytes(b"test")
    vae = tmp_path / "Wan2.2_VAE.pth"
    vae.write_bytes(b"test")

    resolved = resolve_local_edge_assets(edge_snapshot=edge, wan_vae_path=vae)

    assert resolved == {EDGE_REPOSITORY: edge, WAN_VAE_REPOSITORY: vae}


def test_seed_registry_paths_is_repository_specific(tmp_path: Path) -> None:
    edge_path = tmp_path / "edge"
    vae_path = tmp_path / "vae"
    edge_hf = SimpleNamespace(repository=EDGE_REPOSITORY, _path=None)
    vae_hf = SimpleNamespace(repository=WAN_VAE_REPOSITORY, _path=None)
    unrelated_hf = SimpleNamespace(repository="other/repo", _path=None)
    checkpoints = [
        SimpleNamespace(hf=edge_hf),
        SimpleNamespace(hf=vae_hf),
        SimpleNamespace(hf=unrelated_hf),
    ]

    counts = seed_registry_paths(
        checkpoints,
        {EDGE_REPOSITORY: edge_path, WAN_VAE_REPOSITORY: vae_path},
    )

    assert counts == {EDGE_REPOSITORY: 1, WAN_VAE_REPOSITORY: 1}
    assert edge_hf._path == str(edge_path)
    assert vae_hf._path == str(vae_path)
    assert unrelated_hf._path is None


def test_edge_dcp_verifier_allows_absent_source_checkpoint_json(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"weights")
    (model / "config.json").write_text(json.dumps({"model_type": "cosmos3"}), encoding="utf-8")

    assert verify_dcp(tmp_path, require_checkpoint_json=False)["status"] == "passed"
    assert verify_dcp(tmp_path, require_checkpoint_json=True)["status"] == "failed"
