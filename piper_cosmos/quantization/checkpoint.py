"""Create persistent quantized checkpoints independently from inference."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from piper_cosmos.quantization.manifest import (
    QUANTIZATION_MANIFEST,
    QUANTIZATION_SCHEMA_VERSION,
    QuantizationManifest,
)
from piper_cosmos.quantization.registry import get_algorithm, get_backend
from piper_cosmos.quantization.spec import QuantizationSpec


INDEX_FILENAME = "model.safetensors.index.json"
FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
ProgressCallback = Callable[[str], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_layout(checkpoint: Path) -> tuple[dict[str, str], dict[str, Any], Path]:
    index_path = checkpoint / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Expected a sharded Hugging Face checkpoint index at {index_path}. "
            "The initial converter intentionally requires an explicit weight map."
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    parsed = {str(name): str(shard) for name, shard in weight_map.items()}
    for shard in sorted(set(parsed.values())):
        shard_path = checkpoint / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Checkpoint shard is missing: {shard_path}")
    return parsed, payload, index_path


def _selected(spec: QuantizationSpec, name: str, dtype: str, shape: tuple[int, ...]) -> bool:
    return dtype in FLOAT_DTYPES and len(shape) >= 2 and spec.selects_name(name)


def plan_quantized_checkpoint(
    checkpoint: str | Path,
    spec: QuantizationSpec,
) -> dict[str, Any]:
    """Inspect safetensors headers without loading tensor payloads."""

    from safetensors import safe_open

    source = Path(checkpoint).expanduser().resolve()
    spec.validate()
    weight_map, _, index_path = _checkpoint_layout(source)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    selected_count = 0
    skipped_count = 0
    source_bytes = 0
    selected_bytes = 0
    estimated_selected_output_bytes = 0
    for shard, expected_names in sorted(by_shard.items()):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            missing = sorted(set(expected_names) - actual_names)
            if missing:
                raise ValueError(f"{source / shard} is missing indexed tensors: {missing[:5]}")
            for name in expected_names:
                view = handle.get_slice(name)
                shape = tuple(int(value) for value in view.get_shape())
                dtype = view.get_dtype()
                element_size = {
                    "BF16": 2,
                    "F16": 2,
                    "F32": 4,
                    "F64": 8,
                    "I64": 8,
                    "I32": 4,
                    "I16": 2,
                    "I8": 1,
                    "U8": 1,
                    "BOOL": 1,
                }.get(dtype)
                if element_size is None:
                    raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {name}")
                numel = 1
                for dimension in shape:
                    numel *= dimension
                tensor_bytes = numel * element_size
                source_bytes += tensor_bytes
                if _selected(spec, name, dtype, shape):
                    selected_count += 1
                    selected_bytes += tensor_bytes
                    if spec.backend == "cutlass":
                        if len(shape) != 2 or shape[1] % 2:
                            raise ValueError(
                                "The initial CUTLASS packed format requires an "
                                f"even-K 2D tensor, got {name} shape={shape}"
                            )
                        elements = shape[0] * shape[1]
                        estimated_selected_output_bytes += elements // 2
                        estimated_selected_output_bytes += shape[0] * 4
                else:
                    skipped_count += 1

    backend = get_backend(spec.backend)
    return {
        "checkpoint": str(source),
        "index": str(index_path),
        "index_sha256": _sha256(index_path),
        "algo": spec.algo,
        "backend": spec.backend,
        "storage_format": backend.storage_format,
        "bits": int(spec.bits),
        "activation_bits": spec.activation_bits,
        "group_size": int(spec.group_size),
        "weight_granularity": spec.weight_granularity,
        "activation_granularity": spec.activation_granularity,
        "scope": spec.scope,
        "shards": len(by_shard),
        "tensors": selected_count + skipped_count,
        "selected_tensors": selected_count,
        "skipped_tensors": skipped_count,
        "source_tensor_bytes": source_bytes,
        "selected_tensor_bytes": selected_bytes,
        "estimated_output_tensor_bytes": (
            source_bytes
            if spec.backend == "fakequant"
            else source_bytes - selected_bytes + estimated_selected_output_bytes
            if spec.backend == "cutlass"
            else None
        ),
    }


def _copy_checkpoint_metadata(source: Path, destination: Path) -> None:
    for entry in source.iterdir():
        if not entry.is_file():
            continue
        if entry.name == QUANTIZATION_MANIFEST:
            continue
        if entry.name == INDEX_FILENAME or entry.suffix == ".safetensors":
            continue
        shutil.copy2(entry, destination / entry.name)


def create_quantized_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    spec: QuantizationSpec,
    *,
    progress: ProgressCallback | None = None,
) -> QuantizationManifest:
    """Quantize a checkpoint into a new directory and atomically publish it."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    spec.validate()
    if source == destination:
        raise ValueError("Input and output checkpoint directories must be different")
    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}. Choose a new directory; existing checkpoints are never overwritten."
        )

    plan = plan_quantized_checkpoint(source, spec)
    weight_map, index_payload, index_path = _checkpoint_layout(source)
    algorithm = get_algorithm(spec.algo)
    backend = get_backend(spec.backend)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    started = time.perf_counter()
    selected_count = 0
    skipped_count = 0
    selected_names: list[str] = []
    output_weight_map: dict[str, str] = {}
    output_tensor_bytes = 0
    try:
        _copy_checkpoint_metadata(source, temporary)
        total_shards = len(by_shard)
        for shard_index, (shard, names) in enumerate(sorted(by_shard.items()), start=1):
            if progress is not None:
                progress(f"[{shard_index}/{total_shards}] reading {source / shard}")
            output_tensors: dict[str, Any] = {}
            with safe_open(source / shard, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for tensor_index, name in enumerate(names, start=1):
                    tensor = handle.get_tensor(name)
                    if tensor.is_floating_point() and tensor.ndim >= 2 and spec.selects_name(name):
                        encoded = backend.encode_tensor(name, tensor, algorithm, spec)
                        selected_count += 1
                        selected_names.append(name)
                    else:
                        encoded = {name: tensor.contiguous()}
                        skipped_count += 1
                    for output_name, output_tensor in encoded.items():
                        if output_name in output_weight_map:
                            raise ValueError(f"Backend emitted duplicate tensor name: {output_name}")
                        output_tensors[output_name] = output_tensor.contiguous()
                        output_weight_map[output_name] = shard
                        output_tensor_bytes += (
                            int(output_tensor.numel()) * int(output_tensor.element_size())
                        )
                    if progress is not None and tensor_index % 100 == 0:
                        progress(
                            f"[{shard_index}/{total_shards}] processed "
                            f"{tensor_index}/{len(names)} tensors"
                        )
            if progress is not None:
                progress(f"[{shard_index}/{total_shards}] writing {temporary / shard}")
            save_file(output_tensors, temporary / shard, metadata=metadata)
            del output_tensors

        output_index = dict(index_payload)
        output_index["weight_map"] = output_weight_map
        output_metadata = dict(output_index.get("metadata", {}))
        output_metadata["total_size"] = output_tensor_bytes
        output_index["metadata"] = output_metadata
        output_index_path = temporary / INDEX_FILENAME
        output_index_path.write_text(
            json.dumps(output_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        output_bytes = sum(path.stat().st_size for path in temporary.glob("*.safetensors"))
        manifest = QuantizationManifest(
            schema_version=QUANTIZATION_SCHEMA_VERSION,
            quantization=spec,
            storage_format=backend.storage_format,
            source={
                "checkpoint": str(source),
                "index_sha256": _sha256(index_path),
            },
            statistics={
                "shards": len(by_shard),
                "tensors": selected_count + skipped_count,
                "quantized_tensors": selected_count,
                "skipped_tensors": skipped_count,
                "source_tensor_bytes": int(plan["source_tensor_bytes"]),
                "selected_tensor_bytes": int(plan["selected_tensor_bytes"]),
                "output_checkpoint_bytes": int(output_bytes),
                "quantized_tensor_names": selected_names,
                "elapsed_seconds": float(time.perf_counter() - started),
            },
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        (temporary / QUANTIZATION_MANIFEST).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
        if progress is not None:
            progress(f"Published quantized checkpoint: {destination}")
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
