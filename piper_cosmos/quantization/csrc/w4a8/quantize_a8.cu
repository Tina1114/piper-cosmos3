#include "quantize_a8.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void quantize_a8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int cols) {
    const int row = blockIdx.x;
    const auto* input_row = input + static_cast<size_t>(row) * cols;
    auto* output_row = output + static_cast<size_t>(row) * cols;

    float thread_max = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        thread_max = fmaxf(
            thread_max,
            fabsf(__bfloat162float(input_row[col])));
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(
            thread_max,
            __shfl_xor_sync(0xffffffff, thread_max, offset));
    }

    __shared__ float warp_max[32];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        warp_max[warp] = thread_max;
    }
    __syncthreads();

    if (warp == 0) {
        const int warps = (blockDim.x + 31) >> 5;
        float block_max = lane < warps ? warp_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = fmaxf(
                block_max,
                __shfl_xor_sync(0xffffffff, block_max, offset));
        }
        if (lane == 0) {
            // Match the Python reference: an all-zero row uses scale=1.
            warp_max[0] = block_max > 0.0f ? block_max / 127.0f : 1.0f;
            scales[row] = warp_max[0];
        }
    }
    __syncthreads();

    const float inverse_scale = 1.0f / warp_max[0];
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int value = __float2int_rn(
            __bfloat162float(input_row[col]) * inverse_scale);
        value = value < -127 ? -127 : (value > 127 ? 127 : value);
        output_row[col] = static_cast<int8_t>(value);
    }
}

}  // namespace

void piper_quantize_a8_rowwise(
    const __nv_bfloat16* input,
    int8_t* output,
    float* scales,
    int rows,
    int cols,
    cudaStream_t stream) {
    int threads = cols < 256 ? cols : 256;
    threads = ((threads + 31) / 32) * 32;
    threads = threads < 32 ? 32 : threads;
    quantize_a8_rowwise_kernel<<<rows, threads, 0, stream>>>(
        input, output, scales, cols);
}
