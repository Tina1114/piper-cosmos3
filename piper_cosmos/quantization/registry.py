"""Extensible quantization algorithm and runtime backend registries."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from piper_cosmos.quantization.spec import QuantizationSpec


@dataclass
class QuantizedTensor:
    """Backend-neutral quantized representation of one dense tensor."""

    qweight: Any
    scale: Any
    zero_point: Any | None
    original_shape: tuple[int, ...]
    original_dtype: Any
    padded_last_dim: int


class QuantizationAlgorithm(Protocol):
    name: str

    def quantize(self, tensor: Any, spec: QuantizationSpec) -> QuantizedTensor:
        """Quantize one floating-point tensor."""

    def dequantize(self, value: QuantizedTensor, spec: QuantizationSpec) -> Any:
        """Materialize a floating-point tensor from the quantized representation."""


class QuantizationBackend(Protocol):
    name: str
    storage_format: str

    def encode_tensor(
        self,
        name: str,
        tensor: Any,
        algorithm: QuantizationAlgorithm,
        spec: QuantizationSpec,
    ) -> Mapping[str, Any]:
        """Return tensors persisted for one source tensor."""

    def prepare_runtime(self, checkpoint: str, manifest: Mapping[str, Any]) -> str:
        """Prepare a persisted checkpoint for the model loader and return its path."""

    def prepare_model(
        self,
        model: Any,
        manifest: Mapping[str, Any],
        algorithm: QuantizationAlgorithm,
        *,
        checkpoint: str | None = None,
    ) -> Mapping[str, Any]:
        """Install runtime quantized modules after source weights are loaded."""


_ALGORITHMS: dict[str, QuantizationAlgorithm] = {}
_BACKENDS: dict[str, QuantizationBackend] = {}
_BUILTINS_LOADED = False


def register_algorithm(algorithm: QuantizationAlgorithm) -> None:
    name = algorithm.name.strip().lower()
    if not name:
        raise ValueError("Cannot register an algorithm with an empty name")
    if name in _ALGORITHMS:
        raise ValueError(f"Quantization algorithm already registered: {name}")
    _ALGORITHMS[name] = algorithm


def register_backend(backend: QuantizationBackend) -> None:
    name = backend.name.strip().lower()
    if not name:
        raise ValueError("Cannot register a backend with an empty name")
    if name in _BACKENDS:
        raise ValueError(f"Quantization backend already registered: {name}")
    _BACKENDS[name] = backend


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    importlib.import_module("piper_cosmos.quantization.algorithms.rtn")
    importlib.import_module("piper_cosmos.quantization.backends.fakequant")
    importlib.import_module("piper_cosmos.quantization.backends.cutlass")
    _BUILTINS_LOADED = True


def get_algorithm(name: str) -> QuantizationAlgorithm:
    _ensure_builtins()
    key = name.strip().lower()
    try:
        return _ALGORITHMS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown quantization algorithm {name!r}; available={available_algorithms()}"
        ) from exc


def get_backend(name: str) -> QuantizationBackend:
    _ensure_builtins()
    key = name.strip().lower()
    try:
        return _BACKENDS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown quantization backend {name!r}; available={available_backends()}"
        ) from exc


def available_algorithms() -> tuple[str, ...]:
    _ensure_builtins()
    return tuple(sorted(_ALGORITHMS))


def available_backends() -> tuple[str, ...]:
    _ensure_builtins()
    return tuple(sorted(_BACKENDS))
