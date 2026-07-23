"""Validate and prepare persisted quantized checkpoints for inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piper_cosmos.quantization.manifest import QuantizationManifest, load_quantization_manifest
from piper_cosmos.quantization.registry import get_algorithm, get_backend


@dataclass(frozen=True)
class QuantizedCheckpointRuntime:
    checkpoint: str
    active: bool
    manifest: QuantizationManifest | None
    artifact_checkpoint: str | None = None

    def metadata(self) -> dict[str, Any]:
        if self.manifest is None:
            return {"active": False}
        return {
            "active": True,
            "algo": self.manifest.quantization.algo,
            "backend": self.manifest.quantization.backend,
            "bits": self.manifest.quantization.bits,
            "group_size": self.manifest.quantization.group_size,
            "symmetric": self.manifest.quantization.symmetric,
            "activation_bits": self.manifest.quantization.activation_bits,
            "weight_granularity": self.manifest.quantization.weight_granularity,
            "activation_granularity": self.manifest.quantization.activation_granularity,
            "scope": self.manifest.quantization.scope,
            "storage_format": self.manifest.storage_format,
            "manifest": str(
                Path(self.artifact_checkpoint or self.checkpoint)
                / "quantization_config.json"
            ),
            "loader_checkpoint": self.checkpoint,
            "artifact_checkpoint": self.artifact_checkpoint or self.checkpoint,
        }

    def prepare_model(self, model: Any) -> dict[str, Any]:
        if self.manifest is None:
            return {
                "module_type": None,
                "replaced_linear_modules": 0,
                "activation_quantization": False,
            }
        algorithm = get_algorithm(self.manifest.quantization.algo)
        backend = get_backend(self.manifest.quantization.backend)
        return dict(
            backend.prepare_model(
                model,
                self.manifest.to_dict(),
                algorithm,
                checkpoint=self.artifact_checkpoint or self.checkpoint,
            )
        )


def prepare_quantized_checkpoint(
    checkpoint: str | Path,
    *,
    algo: str = "rtn",
    backend: str = "fakequant",
    required: bool = False,
) -> QuantizedCheckpointRuntime:
    path = Path(checkpoint).expanduser()
    manifest = load_quantization_manifest(path, required=required)
    if manifest is None:
        return QuantizedCheckpointRuntime(
            checkpoint=str(path),
            active=False,
            manifest=None,
            artifact_checkpoint=None,
        )

    requested_algo = algo.strip().lower()
    requested_backend = backend.strip().lower()
    actual_algo = manifest.quantization.algo.strip().lower()
    actual_backend = manifest.quantization.backend.strip().lower()
    if requested_algo != actual_algo:
        raise ValueError(
            f"Quantized checkpoint algo mismatch: requested={requested_algo!r}, manifest={actual_algo!r}"
        )
    if requested_backend != actual_backend:
        raise ValueError(
            f"Quantized checkpoint backend mismatch: "
            f"requested={requested_backend!r}, manifest={actual_backend!r}"
        )

    # Resolve both components now so unsupported manifests fail before the
    # heavyweight Cosmos model starts loading.
    get_algorithm(actual_algo)
    runtime_backend = get_backend(actual_backend)
    prepared = runtime_backend.prepare_runtime(str(path), manifest.to_dict())
    return QuantizedCheckpointRuntime(
        checkpoint=str(prepared),
        active=True,
        manifest=manifest,
        artifact_checkpoint=str(path),
    )
