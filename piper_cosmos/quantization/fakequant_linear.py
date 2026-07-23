"""Runtime fake-quant Linear modules used for numerical W4A8 validation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from piper_cosmos.quantization.registry import QuantizationAlgorithm
from piper_cosmos.quantization.spec import QuantizationSpec


class W4A8FakeQuantLinear(nn.Module):
    """RTN W4 output-channel weights with dynamic token-wise A8 activations.

    Values are stored as int8 tensors containing the signed INT4 range. Forward
    dequantizes both operands and calls the regular floating-point Linear kernel,
    so this module validates numerics but does not represent a speed kernel.
    """

    in_features: int
    out_features: int

    def __init__(
        self,
        *,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()
        if qweight.ndim != 2:
            raise ValueError(f"qweight must be [out_features,in_features], got {qweight.shape}")
        if weight_scale.shape != (qweight.shape[0], 1):
            raise ValueError(
                f"weight_scale must be [out_features,1], got {weight_scale.shape}"
            )
        if bias is not None and bias.shape != (qweight.shape[0],):
            raise ValueError(f"bias must be [out_features], got {bias.shape}")

        self.in_features = int(qweight.shape[1])
        self.out_features = int(qweight.shape[0])
        self.activation_bits = int(activation_bits)
        self.register_buffer("qweight", qweight.to(torch.int8).contiguous())
        self.register_buffer("weight_scale", weight_scale.to(torch.float32).contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        algorithm: QuantizationAlgorithm,
        spec: QuantizationSpec,
    ) -> "W4A8FakeQuantLinear":
        spec.validate()
        expected = {
            "bits": 4,
            "activation_bits": 8,
            "weight_granularity": "output_channel",
            "activation_granularity": "token",
            "symmetric": True,
        }
        actual: dict[str, Any] = {
            "bits": spec.bits,
            "activation_bits": spec.activation_bits,
            "weight_granularity": spec.weight_granularity,
            "activation_granularity": spec.activation_granularity,
            "symmetric": spec.symmetric,
        }
        if actual != expected:
            raise ValueError(
                "W4A8FakeQuantLinear requires symmetric W4 output-channel weights "
                f"and token-wise A8 activations; got {actual}"
            )

        source_device = linear.weight.device
        quantized = algorithm.quantize(linear.weight, spec)
        qweight = quantized.qweight.reshape(quantized.original_shape).to(source_device)
        weight_scale = quantized.scale.reshape(linear.out_features, 1).to(source_device)
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(
            qweight=qweight,
            weight_scale=weight_scale,
            bias=bias,
            activation_bits=int(spec.activation_bits),
        )

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        if not activation.is_floating_point():
            raise TypeError(
                f"W4A8FakeQuantLinear expects floating activations, got {activation.dtype}"
            )
        qmax = (1 << (self.activation_bits - 1)) - 1
        work = activation.to(torch.float32)
        activation_scale = work.abs().amax(dim=-1, keepdim=True) / float(qmax)
        activation_scale = torch.where(
            activation_scale > 0,
            activation_scale,
            torch.ones_like(activation_scale),
        )
        activation_q = torch.round(work / activation_scale).clamp(-qmax, qmax)
        activation_dq = (activation_q * activation_scale).to(activation.dtype)

        weight_dq = (self.qweight.to(torch.float32) * self.weight_scale).to(
            activation.dtype
        )
        bias = None if self.bias is None else self.bias.to(activation.dtype)
        return F.linear(activation_dq, weight_dq, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            "weight=W4/output_channel, activation=A8/token"
        )
