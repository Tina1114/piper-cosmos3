#!/usr/bin/env python3
"""Create a persistent quantized Cosmos Piper14 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.quantization import (  # noqa: E402
    QuantizationSpec,
    available_algorithms,
    available_backends,
    create_quantized_checkpoint,
    plan_quantized_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Source HF/safetensors checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="New quantized checkpoint directory.")
    parser.add_argument("--algo", default="rtn", choices=available_algorithms())
    parser.add_argument("--backend", default="fakequant", choices=available_backends())
    parser.add_argument("--bits", "--weight-bits", dest="bits", type=int, default=8)
    parser.add_argument(
        "--activation-bits",
        type=int,
        default=None,
        help="Enable runtime activation fake-quant at this bit width; omit for weight-only.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group along the last weight dimension; 0 means the full row.",
    )
    parser.add_argument("--scope", choices=("all", "generator", "reasoner"), default="all")
    parser.add_argument(
        "--weight-granularity",
        choices=("group", "output_channel"),
        default="group",
    )
    parser.add_argument("--activation-granularity", choices=("token",), default="token")
    parser.add_argument(
        "--include-regex",
        action="append",
        default=[],
        help="Quantize only matching tensor names. Repeat for OR matching.",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=[],
        help="Skip matching tensor names. Repeat for OR matching.",
    )
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use symmetric signed quantization.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect headers and print the plan only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = QuantizationSpec(
        algo=args.algo,
        backend=args.backend,
        bits=args.bits,
        group_size=args.group_size,
        symmetric=args.symmetric,
        activation_bits=args.activation_bits,
        weight_granularity=args.weight_granularity,
        activation_granularity=args.activation_granularity,
        scope=args.scope,
        include_regex=tuple(args.include_regex),
        exclude_regex=tuple(args.exclude_regex),
    ).validate()
    plan = plan_quantized_checkpoint(args.checkpoint, spec)
    print(json.dumps({"plan": plan, "quantization": spec.to_dict()}, indent=2))
    if args.dry_run:
        return
    manifest = create_quantized_checkpoint(
        args.checkpoint,
        args.output,
        spec,
        progress=lambda message: print(f"[quantize] {message}", flush=True),
    )
    print(json.dumps({"output": str(args.output.resolve()), "manifest": manifest.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
