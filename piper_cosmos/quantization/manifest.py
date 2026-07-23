"""Manifest stored next to a quantized Cosmos checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from piper_cosmos.quantization.spec import QuantizationSpec


QUANTIZATION_MANIFEST = "quantization_config.json"
QUANTIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QuantizationManifest:
    schema_version: int
    quantization: QuantizationSpec
    storage_format: str
    source: dict[str, Any]
    statistics: dict[str, Any]
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quantization"] = self.quantization.to_dict()
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QuantizationManifest":
        schema_version = int(payload.get("schema_version", -1))
        if schema_version != QUANTIZATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported quantization manifest schema {schema_version}; "
                f"expected {QUANTIZATION_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=schema_version,
            quantization=QuantizationSpec.from_mapping(payload["quantization"]),
            storage_format=str(payload["storage_format"]),
            source=dict(payload.get("source", {})),
            statistics=dict(payload.get("statistics", {})),
            created_at_utc=str(payload.get("created_at_utc", "")),
        )


def load_quantization_manifest(
    checkpoint: str | Path,
    *,
    required: bool = False,
) -> QuantizationManifest | None:
    path = Path(checkpoint).expanduser() / QUANTIZATION_MANIFEST
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Quantization manifest not found: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Quantization manifest must contain a JSON object: {path}")
    return QuantizationManifest.from_mapping(payload)
