"""Fake-quant backend that persists dequantized tensors in standard safetensors."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from piper_cosmos.quantization.fakequant_linear import W4A8FakeQuantLinear
from piper_cosmos.quantization.registry import (
    QuantizationAlgorithm,
    register_backend,
)
from piper_cosmos.quantization.spec import QuantizationSpec


class FakeQuantBackend:
    """Quantize then dequantize weights and use the unmodified Cosmos kernels."""

    name = "fakequant"
    storage_format = "hf-safetensors-fakequant-v1"

    def encode_tensor(
        self,
        name: str,
        tensor: Any,
        algorithm: QuantizationAlgorithm,
        spec: QuantizationSpec,
    ) -> Mapping[str, Any]:
        quantized = algorithm.quantize(tensor, spec)
        return {name: algorithm.dequantize(quantized, spec)}

    def prepare_runtime(self, checkpoint: str, manifest: Mapping[str, Any]) -> str:
        storage_format = manifest.get("storage_format")
        if storage_format != self.storage_format:
            raise ValueError(
                f"fakequant expected storage_format={self.storage_format!r}, got {storage_format!r}"
            )
        return checkpoint

    def prepare_model(
        self,
        model: Any,
        manifest: Mapping[str, Any],
        algorithm: QuantizationAlgorithm,
        *,
        checkpoint: str | None = None,
    ) -> Mapping[str, Any]:
        del checkpoint
        spec = QuantizationSpec.from_mapping(manifest["quantization"])
        if spec.activation_bits is None:
            return {
                "module_type": None,
                "replaced_linear_modules": 0,
                "activation_quantization": False,
            }

        def normalize(name: str) -> str:
            return (
                name.removeprefix("model.")
                .replace("._orig_mod.", ".")
                .replace("._checkpoint_wrapped_module.", ".")
            )

        selected_names = {
            normalize(str(name))
            for name in manifest.get("statistics", {}).get("quantized_tensor_names", [])
        }
        replacements: list[tuple[str, nn.Linear]] = []
        for module_name, module in list(model.named_modules()):
            if not module_name or not isinstance(module, nn.Linear):
                continue
            normalized_weight_name = normalize(f"{module_name}.weight")
            if selected_names:
                if normalized_weight_name not in selected_names:
                    continue
            elif not spec.selects_name(normalized_weight_name):
                continue
            replacements.append((module_name, module))

        for module_name, module in replacements:
            replacement = W4A8FakeQuantLinear.from_linear(
                module,
                algorithm=algorithm,
                spec=spec,
            )
            parent_name, _, child_name = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            parent._modules[child_name] = replacement

        return {
            "module_type": "W4A8FakeQuantLinear",
            "replaced_linear_modules": len(replacements),
            "activation_quantization": True,
            "weight_granularity": spec.weight_granularity,
            "activation_granularity": spec.activation_granularity,
        }


register_backend(FakeQuantBackend())
