#!/usr/bin/env python3
"""Microbenchmark the first packed-W4/CUTLASS kernel on Cosmos GEMM shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.quantization.packed_linear import (  # noqa: E402
    load_cutlass_extension,
    pack_int4,
)


SHAPES = (
    ("q_o_proj", 4096, 4096),
    ("k_v_proj", 1024, 4096),
    ("gate_up_proj", 12288, 4096),
    ("down_proj", 4096, 12288),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=3233, help="Flattened M dimension.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def elapsed_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    extension = load_cutlass_extension()
    torch.manual_seed(args.seed)
    results = []
    for label, output_features, input_features in SHAPES:
        activation = torch.randn(
            args.tokens,
            input_features,
            device="cuda",
            dtype=torch.bfloat16,
        )
        qweight = torch.randint(
            -7,
            8,
            (output_features, input_features),
            device="cuda",
            dtype=torch.int8,
        )
        packed_qweight = pack_int4(qweight)
        del qweight
        weight_scale = (
            torch.rand(output_features, device="cuda", dtype=torch.float32)
            * 0.02
            + 0.001
        )

        def invoke(
            activation_value: torch.Tensor = activation,
            packed_value: torch.Tensor = packed_qweight,
            scale_value: torch.Tensor = weight_scale,
        ) -> torch.Tensor:
            return torch.ops.piper_w4a8.linear(
                activation_value,
                packed_value,
                scale_value,
                None,
            )

        def invoke_debug(
            activation_value: torch.Tensor = activation,
            packed_value: torch.Tensor = packed_qweight,
            scale_value: torch.Tensor = weight_scale,
        ) -> torch.Tensor:
            return torch.ops.piper_w4a8.linear_debug(
                activation_value,
                packed_value,
                scale_value,
                None,
            )

        duration_ms = elapsed_ms(invoke, args.warmup, args.iterations)
        debug_duration_ms = elapsed_ms(
            invoke_debug,
            args.warmup,
            args.iterations,
        )
        integer_ops = 2 * args.tokens * output_features * input_features
        results.append(
            {
                "name": label,
                "M": args.tokens,
                "N": output_features,
                "K": input_features,
                "milliseconds": duration_ms,
                "debug_unpack_milliseconds": debug_duration_ms,
                "speedup_over_debug": debug_duration_ms / duration_ms,
                "effective_tops": integer_ops / (duration_ms * 1e9),
            }
        )
        del activation, packed_qweight, weight_scale
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "compute_capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "kernel": extension.kernel_variant(),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
