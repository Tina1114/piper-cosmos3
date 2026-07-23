#!/usr/bin/env python3
"""Build the optional Piper packed-W4/CUTLASS CUDA extension in place."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "piper_cosmos" / "quantization" / "csrc" / "w4a8"
CUTLASS_DIR = Path(
    os.environ.get("PIPER_CUTLASS_DIR", ROOT / "third_party" / "cutlass")
).expanduser().resolve()
CUTLASS_INCLUDE = CUTLASS_DIR / "include"
CUTLASS_UTIL_INCLUDE = CUTLASS_DIR / "tools" / "util" / "include"

if not (CUTLASS_INCLUDE / "cutlass" / "cutlass.h").is_file():
    raise RuntimeError(
        f"CUTLASS headers not found under {CUTLASS_INCLUDE}. "
        "Set PIPER_CUTLASS_DIR to a pinned CUTLASS checkout (tested: v4.4.2)."
    )

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

extension = CUDAExtension(
    name="piper_cosmos.quantization._w4a8_cuda",
    sources=[
        str(SOURCE_DIR / "bindings.cpp"),
        str(SOURCE_DIR / "quantize_a8.cu"),
        str(SOURCE_DIR / "unpack_int4.cu"),
        str(SOURCE_DIR / "cutlass_w8a8.cu"),
        str(SOURCE_DIR / "cutlass_w4a8_fused.cu"),
    ],
    include_dirs=[
        str(SOURCE_DIR),
        str(CUTLASS_INCLUDE),
        str(CUTLASS_UTIL_INCLUDE),
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-std=c++17",
        ],
    },
)

setup(
    name="piper-cosmos-quant-kernels",
    version="0.1.0",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
