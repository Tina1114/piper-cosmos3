"""Packed W4A8 Linear module and reference helpers.

The production forward path is intentionally strict: it requires the compiled
CUDA/CUTLASS extension.  The explicit ``reference_forward`` method exists for
checkpoint and numerical tests, but is never selected silently at runtime.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from piper_cosmos.quantization.registry import QuantizationAlgorithm
from piper_cosmos.quantization.spec import QuantizationSpec


def validate_w4a8_cutlass_spec(spec: QuantizationSpec) -> None:
    """Validate the narrow contract implemented by the first CUTLASS backend."""

    spec.validate()
    expected: dict[str, Any] = {
        "bits": 4,
        "activation_bits": 8,
        "weight_granularity": "output_channel",
        "activation_granularity": "token",
        "symmetric": True,
        "group_size": 0,
    }
    actual: dict[str, Any] = {
        "bits": spec.bits,
        "activation_bits": spec.activation_bits,
        "weight_granularity": spec.weight_granularity,
        "activation_granularity": spec.activation_granularity,
        "symmetric": spec.symmetric,
        "group_size": spec.group_size,
    }
    if actual != expected:
        raise ValueError(
            "The initial CUTLASS backend requires symmetric W4 output-channel "
            f"weights and token-wise A8 activations; expected={expected}, got={actual}"
        )


def pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values along K, low nibble first.

    Logical input is ``[N, K]``.  Physical output is UINT8 ``[N, K / 2]``:
    ``low=W[n,2*j]`` and ``high=W[n,2*j+1]`` in two's-complement nibble form.
    """

    if qweight.ndim != 2:
        raise ValueError(f"qweight must be [N,K], got shape={tuple(qweight.shape)}")
    if qweight.shape[1] % 2:
        raise ValueError(f"INT4 packing requires an even K, got K={qweight.shape[1]}")
    if qweight.numel():
        qmin = int(qweight.min())
        qmax = int(qweight.max())
        if qmin < -8 or qmax > 7:
            raise ValueError(f"signed INT4 values must be in [-8,7], got [{qmin},{qmax}]")

    work = qweight.to(torch.int16)
    low = torch.bitwise_and(work[:, 0::2], 0xF)
    high = torch.bitwise_left_shift(torch.bitwise_and(work[:, 1::2], 0xF), 4)
    return torch.bitwise_or(low, high).to(torch.uint8).contiguous()


def unpack_int4(packed_qweight: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    """Reference unpack for tests and explicit debug execution."""

    if packed_qweight.ndim != 2 or packed_qweight.dtype != torch.uint8:
        raise ValueError(
            "packed_qweight must be a UINT8 [N,packed_K] tensor, "
            f"got dtype={packed_qweight.dtype}, shape={tuple(packed_qweight.shape)}"
        )
    physical_k = int(packed_qweight.shape[1]) * 2
    logical_k = physical_k if in_features is None else int(in_features)
    if logical_k <= 0 or logical_k > physical_k:
        raise ValueError(
            f"in_features must be in [1,{physical_k}], got {logical_k}"
        )

    low = torch.bitwise_and(packed_qweight, 0xF).to(torch.int8)
    high = torch.bitwise_right_shift(packed_qweight, 4).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    output = torch.empty(
        (packed_qweight.shape[0], physical_k),
        dtype=torch.int8,
        device=packed_qweight.device,
    )
    output[:, 0::2] = low
    output[:, 1::2] = high
    return output[:, :logical_k].contiguous()


def quantize_activation_tokenwise(
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference symmetric token-wise dynamic A8 quantization."""

    if not activation.is_floating_point():
        raise TypeError(f"activation must be floating point, got {activation.dtype}")
    work = activation.to(torch.float32)
    scale = work.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(work / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


@lru_cache(maxsize=1)
def load_cutlass_extension() -> Any:
    """Import the AOT extension and make its torch.library operators visible."""

    try:
        return importlib.import_module("piper_cosmos.quantization._w4a8_cuda")
    except Exception as exc:
        raise RuntimeError(
            "The CUTLASS W4A8 extension is not available. Build it with "
            "`PIPER_CUTLASS_DIR=/path/to/cutlass "
            "python setup_quant_kernels.py build_ext --inplace`. "
            "The CUDA toolkit used by nvcc must match the CUDA major version "
            "of PyTorch."
        ) from exc


class W4A8CutlassLinear(nn.Module):
    """Packed W4 weights with dynamic token-wise A8 CUTLASS inference."""

    in_features: int
    out_features: int

    def __init__(
        self,
        *,
        packed_qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        in_features: int,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if packed_qweight.ndim != 2 or packed_qweight.dtype != torch.uint8:
            raise ValueError("packed_qweight must be UINT8 [out_features,packed_K]")
        if weight_scale.numel() != packed_qweight.shape[0]:
            raise ValueError("weight_scale must have one value per output channel")
        if int(in_features) != packed_qweight.shape[1] * 2:
            raise ValueError(
                "The first kernel requires even K with no physical padding: "
                f"in_features={in_features}, packed_K={packed_qweight.shape[1]}"
            )
        if bias is not None and bias.numel() != packed_qweight.shape[0]:
            raise ValueError("bias must have one value per output channel")

        self.in_features = int(in_features)
        self.out_features = int(packed_qweight.shape[0])
        self.register_buffer("packed_qweight", packed_qweight.contiguous())
        self.register_buffer(
            "weight_scale", weight_scale.reshape(-1).to(torch.float32).contiguous()
        )
        self.register_buffer("weight_zero", None)
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        algorithm: QuantizationAlgorithm,
        spec: QuantizationSpec,
    ) -> "W4A8CutlassLinear":
        validate_w4a8_cutlass_spec(spec)
        quantized = algorithm.quantize(linear.weight, spec)
        qweight = quantized.qweight.reshape(quantized.original_shape)
        return cls(
            packed_qweight=pack_int4(qweight).to(linear.weight.device),
            weight_scale=quantized.scale.reshape(-1).to(linear.weight.device),
            in_features=linear.in_features,
            bias=None if linear.bias is None else linear.bias.detach(),
        )

    def reference_forward(self, activation: torch.Tensor) -> torch.Tensor:
        """Explicit slow reference; production ``forward`` never calls this."""

        activation_q, activation_scale = quantize_activation_tokenwise(activation)
        activation_dq = (activation_q.to(torch.float32) * activation_scale).to(
            activation.dtype
        )
        qweight = unpack_int4(self.packed_qweight, self.in_features)
        weight = (qweight.to(torch.float32) * self.weight_scale[:, None]).to(
            activation.dtype
        )
        bias = None if self.bias is None else self.bias.to(activation.dtype)
        return F.linear(activation_dq, weight, bias)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        load_cutlass_extension()
        if not activation.is_cuda:
            raise RuntimeError("W4A8CutlassLinear requires a CUDA activation")
        if activation.dtype != torch.bfloat16:
            raise TypeError(
                "The initial W4A8 CUTLASS kernel requires bfloat16 activation, "
                f"got {activation.dtype}"
            )
        if activation.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected activation K={self.in_features}, got {activation.shape[-1]}"
            )

        leading_shape = tuple(activation.shape[:-1])
        activation_2d = activation.reshape(-1, self.in_features).contiguous()
        output_2d = torch.ops.piper_w4a8.linear(
            activation_2d,
            self.packed_qweight,
            self.weight_scale,
            self.bias,
        )
        return output_2d.reshape(*leading_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            "weight=packed-W4/output_channel, activation=A8/token, "
            "kernel=cutlass-fused-mainloop"
        )
