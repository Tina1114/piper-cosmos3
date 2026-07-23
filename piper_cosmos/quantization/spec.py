"""Serializable quantization configuration."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


VALID_SCOPES = ("all", "generator", "reasoner")
VALID_WEIGHT_GRANULARITIES = ("group", "output_channel")
VALID_ACTIVATION_GRANULARITIES = ("token",)


@dataclass(frozen=True)
class QuantizationSpec:
    """Algorithm/backend-independent checkpoint quantization options."""

    algo: str = "rtn"
    backend: str = "fakequant"
    bits: int = 8
    group_size: int = 128
    symmetric: bool = True
    activation_bits: int | None = None
    weight_granularity: str = "group"
    activation_granularity: str = "token"
    scope: str = "all"
    include_regex: tuple[str, ...] = field(default_factory=tuple)
    exclude_regex: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> "QuantizationSpec":
        if not self.algo.strip():
            raise ValueError("Quantization algo must be non-empty")
        if not self.backend.strip():
            raise ValueError("Quantization backend must be non-empty")
        if not 2 <= int(self.bits) <= 8:
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")
        if int(self.group_size) < 0:
            raise ValueError(f"group_size must be >= 0, got {self.group_size}")
        if self.activation_bits is not None and not 2 <= int(self.activation_bits) <= 8:
            raise ValueError(
                f"activation_bits must be None or in [2, 8], got {self.activation_bits}"
            )
        if self.weight_granularity not in VALID_WEIGHT_GRANULARITIES:
            raise ValueError(
                "weight_granularity must be one of "
                f"{VALID_WEIGHT_GRANULARITIES}, got {self.weight_granularity!r}"
            )
        if self.weight_granularity == "output_channel" and int(self.group_size) != 0:
            raise ValueError(
                "output_channel weight granularity requires group_size=0 "
                "(one scale for each output row)"
            )
        if self.activation_granularity not in VALID_ACTIVATION_GRANULARITIES:
            raise ValueError(
                "activation_granularity must be one of "
                f"{VALID_ACTIVATION_GRANULARITIES}, got {self.activation_granularity!r}"
            )
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {VALID_SCOPES}, got {self.scope!r}")
        for pattern in (*self.include_regex, *self.exclude_regex):
            re.compile(pattern)
        return self

    def selects_name(self, name: str) -> bool:
        if self.scope == "generator" and "_moe_gen" not in name:
            return False
        if self.scope == "reasoner" and not (
            "language_model" in name and "_moe_gen" not in name
        ):
            return False
        if self.include_regex and not any(re.search(pattern, name) for pattern in self.include_regex):
            return False
        if any(re.search(pattern, name) for pattern in self.exclude_regex):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["include_regex"] = list(self.include_regex)
        payload["exclude_regex"] = list(self.exclude_regex)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QuantizationSpec":
        values = dict(payload)
        # Manifests created before activation fake-quant was introduced were
        # weight-only. Preserve their behavior rather than silently enabling A8.
        values.setdefault("activation_bits", None)
        values.setdefault("weight_granularity", "group")
        values.setdefault("activation_granularity", "token")
        values["include_regex"] = tuple(values.get("include_regex", ()))
        values["exclude_regex"] = tuple(values.get("exclude_regex", ()))
        return cls(**values).validate()
