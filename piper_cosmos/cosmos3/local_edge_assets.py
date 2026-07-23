"""Strictly local asset bootstrap for Cosmos3-Edge Piper14 workflows."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EDGE_REPOSITORY = "nvidia/Cosmos3-Edge"
WAN_VAE_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
ENV_EDGE_SNAPSHOT = "COSMOS3_EDGE_SNAPSHOT"
ENV_WAN_VAE_PATH = "COSMOS3_WAN_VAE_PATH"

DEFAULT_EDGE_SNAPSHOT = REPO_ROOT / "external" / "cosmos" / "checkpoints" / "Cosmos3-Edge"
DEFAULT_WAN_VAE_PATH = Path(
    "/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/"
    "models--Wan-AI--Wan2.2-TI2V-5B/snapshots/"
    "921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
REQUIRED_EDGE_FILES = (
    "config.json",
    "model_index.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
)

_LOCAL_REPOSITORY_PATHS: dict[str, Path] = {}
_ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD = None
_ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD = None


def _resolve_path(path: Path | None, env_var: str, default: Path) -> Path:
    if path is not None:
        return path
    raw = os.environ.get(env_var, "").strip()
    return Path(raw) if raw else default


def set_offline_runtime_env(edge_snapshot: Path, wan_vae_path: Path) -> None:
    cache_root = REPO_ROOT / "external" / "cosmos" / "cache"
    os.environ.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    os.environ.setdefault("UV_TOOL_DIR", str(cache_root / "uv" / "tools"))
    os.environ.setdefault("XDG_DATA_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "external" / "cosmos" / "checkpoints" / "hf_home"))
    os.environ.setdefault("IMAGINAIRE_CACHE_DIR", str(cache_root / "imaginaire"))
    os.environ[ENV_EDGE_SNAPSHOT] = str(edge_snapshot)
    os.environ[ENV_WAN_VAE_PATH] = str(wan_vae_path)
    os.environ["WAN_VAE_PATH"] = str(wan_vae_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_local_edge_assets(
    *,
    edge_snapshot: Path | None = None,
    wan_vae_path: Path | None = None,
) -> dict[str, Path]:
    edge_snapshot = _resolve_path(edge_snapshot, ENV_EDGE_SNAPSHOT, DEFAULT_EDGE_SNAPSHOT)
    wan_vae_path = _resolve_path(wan_vae_path, ENV_WAN_VAE_PATH, DEFAULT_WAN_VAE_PATH)

    missing: list[str] = []
    if not edge_snapshot.is_dir():
        missing.append(f"missing Cosmos3-Edge snapshot directory: {edge_snapshot}")
    else:
        missing.extend(str(edge_snapshot / relpath) for relpath in REQUIRED_EDGE_FILES if not (edge_snapshot / relpath).is_file())
        if not list(edge_snapshot.glob("*.safetensors")) and not list((edge_snapshot / "transformer").glob("*.safetensors")):
            missing.append(f"missing Cosmos3-Edge safetensors shards under: {edge_snapshot}")
    if not wan_vae_path.is_file():
        missing.append(f"missing Wan VAE asset: {wan_vae_path}")
    if missing:
        raise FileNotFoundError("offline asset validation failed: " + "; ".join(missing))

    return {EDGE_REPOSITORY: edge_snapshot, WAN_VAE_REPOSITORY: wan_vae_path}


def seed_registry_paths(
    checkpoints: Iterable[object],
    repository_paths: Mapping[str, Path],
) -> dict[str, int]:
    seeded = {repository: 0 for repository in repository_paths}
    for checkpoint in checkpoints:
        hf = getattr(checkpoint, "hf", None)
        repository = getattr(hf, "repository", None)
        if repository not in repository_paths:
            continue
        hf._path = str(repository_paths[repository])
        seeded[repository] += 1
    return seeded


def _local_checkpoint_dir_hf_download(checkpoint: object) -> str:
    repository = getattr(checkpoint, "repository", "")
    if repository in _LOCAL_REPOSITORY_PATHS:
        path = _LOCAL_REPOSITORY_PATHS[repository]
        subdirectory = getattr(checkpoint, "subdirectory", "")
        return str(path / subdirectory) if subdirectory else str(path)
    if _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD is None:
        raise RuntimeError("original CheckpointDirHf._download is unavailable")
    return _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD(checkpoint)


def _local_checkpoint_file_hf_download(checkpoint: object) -> str:
    repository = getattr(checkpoint, "repository", "")
    if repository == WAN_VAE_REPOSITORY:
        return str(_LOCAL_REPOSITORY_PATHS[WAN_VAE_REPOSITORY])
    if _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD is None:
        raise RuntimeError("original CheckpointFileHf._download is unavailable")
    return _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD(checkpoint)


def patch_hf_downloads(repository_paths: Mapping[str, Path]) -> None:
    from cosmos_framework.utils.checkpoint_db import CheckpointDirHf, CheckpointFileHf

    global _LOCAL_REPOSITORY_PATHS
    global _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD
    global _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD

    _LOCAL_REPOSITORY_PATHS = dict(repository_paths)
    if _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD is None:
        _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD = CheckpointDirHf._download
        CheckpointDirHf._download = _local_checkpoint_dir_hf_download
    if _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD is None:
        _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD = CheckpointFileHf._download
        CheckpointFileHf._download = _local_checkpoint_file_hf_download


def bootstrap_local_edge_assets(
    *,
    edge_snapshot: Path | None = None,
    wan_vae_path: Path | None = None,
) -> dict[str, Path]:
    repository_paths = resolve_local_edge_assets(edge_snapshot=edge_snapshot, wan_vae_path=wan_vae_path)
    set_offline_runtime_env(repository_paths[EDGE_REPOSITORY], repository_paths[WAN_VAE_REPOSITORY])

    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

    register_checkpoints()
    seeded = seed_registry_paths(_CHECKPOINTS.values(), repository_paths)
    if seeded[EDGE_REPOSITORY] == 0:
        raise RuntimeError(f"No checkpoint registry entries were seeded for {EDGE_REPOSITORY}.")
    if seeded[WAN_VAE_REPOSITORY] == 0:
        raise RuntimeError(f"No checkpoint registry entries were seeded for {WAN_VAE_REPOSITORY}.")
    patch_hf_downloads(repository_paths)
    return repository_paths
