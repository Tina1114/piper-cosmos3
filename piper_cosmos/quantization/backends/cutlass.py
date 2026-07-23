"""Packed W4A8 checkpoint backend backed by the CUDA/CUTLASS extension."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from piper_cosmos.quantization.packed_linear import (
    W4A8CutlassLinear,
    pack_int4,
    validate_w4a8_cutlass_spec,
)
from piper_cosmos.quantization.registry import (
    QuantizationAlgorithm,
    register_backend,
)
from piper_cosmos.quantization.spec import QuantizationSpec


INDEX_FILENAME = "model.safetensors.index.json"


def packed_tensor_names(weight_name: str) -> tuple[str, str]:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a .weight tensor name, got {weight_name!r}")
    stem = weight_name[: -len(".weight")]
    return f"{stem}.qweight", f"{stem}.weight_scale"


def _normalize_weight_name(name: str) -> str:
    return (
        name.removeprefix("model.")
        .replace("._orig_mod.", ".")
        .replace("._checkpoint_wrapped_module.", ".")
    )


class CutlassBackend:
    """Persist packed W4 and install a strict CUTLASS runtime module.

    The first implementation is a sidecar runtime: Cosmos loads the original
    checkpoint, then selected Linear modules are replaced from this packed
    artifact.  It proves the storage format and real integer kernel without
    requiring invasive changes to the upstream Cosmos loader.
    """

    name = "cutlass"
    storage_format = "cosmos-packed-w4a8-cutlass-v1"

    def encode_tensor(
        self,
        name: str,
        tensor: Any,
        algorithm: QuantizationAlgorithm,
        spec: QuantizationSpec,
    ) -> Mapping[str, Any]:
        validate_w4a8_cutlass_spec(spec)
        if not name.endswith(".weight") or tensor.ndim != 2:
            raise ValueError(
                "The initial CUTLASS backend only encodes 2D .weight tensors, "
                f"got name={name!r}, shape={tuple(tensor.shape)}"
            )
        quantized = algorithm.quantize(tensor, spec)
        if quantized.padded_last_dim != quantized.original_shape[-1]:
            raise ValueError("The first packed format does not support physical K padding")
        qweight = quantized.qweight.reshape(quantized.original_shape)
        packed_name, scale_name = packed_tensor_names(name)
        return {
            packed_name: pack_int4(qweight).cpu(),
            scale_name: quantized.scale.reshape(-1).to(torch.float32).contiguous(),
        }

    def prepare_runtime(self, checkpoint: str, manifest: Mapping[str, Any]) -> str:
        storage_format = manifest.get("storage_format")
        if storage_format != self.storage_format:
            raise ValueError(
                f"cutlass expected storage_format={self.storage_format!r}, "
                f"got {storage_format!r}"
            )
        source = Path(str(manifest.get("source", {}).get("checkpoint", ""))).expanduser()
        if not source.is_dir():
            raise FileNotFoundError(
                "The first CUTLASS sidecar runtime still needs the original "
                f"checkpoint recorded by the manifest; not found: {source}"
            )
        artifact = Path(checkpoint).expanduser()
        if not (artifact / INDEX_FILENAME).is_file():
            raise FileNotFoundError(f"Packed checkpoint index not found: {artifact / INDEX_FILENAME}")
        return str(source)

    def prepare_model(
        self,
        model: Any,
        manifest: Mapping[str, Any],
        algorithm: QuantizationAlgorithm,
        *,
        checkpoint: str | None = None,
    ) -> Mapping[str, Any]:
        del algorithm
        if checkpoint is None:
            raise ValueError("CUTLASS prepare_model requires the packed artifact checkpoint")
        spec = QuantizationSpec.from_mapping(manifest["quantization"])
        validate_w4a8_cutlass_spec(spec)

        selected = {
            _normalize_weight_name(str(name)): str(name)
            for name in manifest.get("statistics", {}).get("quantized_tensor_names", [])
        }
        # Store names only. Holding nn.Linear objects here keeps every original
        # BF16 weight alive until the entire replacement pass finishes, causing
        # the packed weights to stack on top of the full-precision Generator.
        replacements: dict[str, tuple[str, str]] = {}
        for module_name, module in model.named_modules():
            if not module_name or not isinstance(module, nn.Linear):
                continue
            normalized = _normalize_weight_name(f"{module_name}.weight")
            original_name = selected.get(normalized)
            if original_name is None:
                continue
            packed_name, _ = packed_tensor_names(original_name)
            replacements[packed_name] = (module_name, original_name)
        # The loop variable otherwise keeps the final visited module alive.
        del module

        artifact = Path(checkpoint).expanduser()
        index = json.loads((artifact / INDEX_FILENAME).read_text(encoding="utf-8"))
        weight_map = {str(k): str(v) for k, v in index["weight_map"].items()}
        by_shard: dict[str, list[str]] = defaultdict(list)
        for packed_name in replacements:
            scale_name = packed_tensor_names(replacements[packed_name][1])[1]
            packed_shard = weight_map.get(packed_name)
            scale_shard = weight_map.get(scale_name)
            if packed_shard is None or scale_shard is None:
                raise KeyError(
                    f"Packed checkpoint is missing {packed_name!r} or {scale_name!r}"
                )
            if packed_shard != scale_shard:
                raise ValueError(
                    f"Packed weight and scale must share a shard: {packed_name}"
                )
            by_shard[packed_shard].append(packed_name)

        from safetensors import safe_open

        replaced = 0
        with torch.no_grad():
            for shard, packed_names in sorted(by_shard.items()):
                with safe_open(artifact / shard, framework="pt", device="cpu") as handle:
                    for packed_name in packed_names:
                        module_name, original_name = replacements.pop(packed_name)
                        _, scale_name = packed_tensor_names(original_name)
                        source_module = model.get_submodule(module_name)
                        if not isinstance(source_module, nn.Linear):
                            raise TypeError(
                                f"Expected nn.Linear at {module_name!r}, "
                                f"got {type(source_module).__name__}"
                            )
                        # Read the packed CPU payload before mutating the model.
                        packed_cpu = handle.get_tensor(packed_name)
                        scale_cpu = handle.get_tensor(scale_name)

                        device = source_module.weight.device
                        in_features = source_module.in_features
                        bias = source_module.bias
                        parent_name, _, child_name = module_name.rpartition(".")
                        parent = model.get_submodule(parent_name) if parent_name else model

                        # Unlink the BF16 module before allocating its packed
                        # replacement on CUDA. Refcounting immediately returns
                        # the large weight block to PyTorch's caching allocator,
                        # so the following .to(device) can reuse that block.
                        parent._modules[child_name] = nn.Identity()
                        del source_module

                        packed = packed_cpu.to(device)
                        scale = scale_cpu.to(device)
                        replacement = W4A8CutlassLinear(
                            packed_qweight=packed,
                            weight_scale=scale,
                            in_features=in_features,
                            bias=bias,
                        )
                        parent._modules[child_name] = replacement
                        replaced += 1

        return {
            "module_type": "W4A8CutlassLinear",
            "replaced_linear_modules": replaced,
            "activation_quantization": True,
            "weight_packing": "signed-int4-low-nibble-first",
            "kernel": "cutlass-w4a8-fused-mainloop",
            "standalone_checkpoint": False,
        }


register_backend(CutlassBackend())
