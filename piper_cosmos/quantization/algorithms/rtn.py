"""Symmetric grouped round-to-nearest (RTN) weight quantization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from piper_cosmos.quantization.registry import QuantizedTensor, register_algorithm
from piper_cosmos.quantization.spec import QuantizationSpec


class RTNAlgorithm:
    name = "rtn"

    def quantize(self, tensor: torch.Tensor, spec: QuantizationSpec) -> QuantizedTensor:
        spec.validate()
        if not tensor.is_floating_point():
            raise TypeError(f"RTN expects a floating tensor, got {tensor.dtype}")
        if tensor.ndim < 2:
            raise ValueError(f"RTN expects tensor.ndim >= 2, got shape={tuple(tensor.shape)}")
        if not spec.symmetric:
            raise NotImplementedError("The initial RTN implementation supports symmetric quantization only")

        original_shape = tuple(int(value) for value in tensor.shape)
        last_dim = original_shape[-1]
        group_size = last_dim if spec.weight_granularity == "output_channel" else int(spec.group_size) or last_dim
        group_size = min(group_size, last_dim)
        padded_last_dim = ((last_dim + group_size - 1) // group_size) * group_size

        work = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1, last_dim)
        if padded_last_dim != last_dim:
            work = F.pad(work, (0, padded_last_dim - last_dim))
        grouped = work.reshape(work.shape[0], padded_last_dim // group_size, group_size)

        qmax = (1 << (int(spec.bits) - 1)) - 1
        scale = grouped.abs().amax(dim=-1, keepdim=True) / float(qmax)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight = torch.round(grouped / scale).clamp(-qmax, qmax).to(torch.int8)
        return QuantizedTensor(
            qweight=qweight,
            scale=scale,
            zero_point=None,
            original_shape=original_shape,
            original_dtype=tensor.dtype,
            padded_last_dim=padded_last_dim,
        )

    def dequantize(self, value: QuantizedTensor, spec: QuantizationSpec) -> torch.Tensor:
        del spec
        last_dim = value.original_shape[-1]
        restored = value.qweight.to(torch.float32) * value.scale
        restored = restored.reshape(-1, value.padded_last_dim)[:, :last_dim]
        return restored.reshape(value.original_shape).to(value.original_dtype).contiguous()


register_algorithm(RTNAlgorithm())
