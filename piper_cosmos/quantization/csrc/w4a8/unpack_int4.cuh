#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void piper_unpack_int4(
    const uint8_t* packed,
    int8_t* unpacked,
    int rows,
    int packed_cols,
    cudaStream_t stream);
