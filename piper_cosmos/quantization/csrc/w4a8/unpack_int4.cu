#include "unpack_int4.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ int8_t sign_extend_int4(uint8_t value) {
    value &= 0xF;
    return static_cast<int8_t>(value < 8 ? value : static_cast<int>(value) - 16);
}

__global__ void unpack_int4_kernel(
    const uint8_t* __restrict__ packed,
    int8_t* __restrict__ unpacked,
    size_t packed_elements) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= packed_elements) {
        return;
    }
    const uint8_t value = packed[index];
    unpacked[index * 2] = sign_extend_int4(value);
    unpacked[index * 2 + 1] = sign_extend_int4(value >> 4);
}

}  // namespace

void piper_unpack_int4(
    const uint8_t* packed,
    int8_t* unpacked,
    int rows,
    int packed_cols,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const size_t elements = static_cast<size_t>(rows) * packed_cols;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    unpack_int4_kernel<<<blocks, threads, 0, stream>>>(
        packed, unpacked, elements);
}
