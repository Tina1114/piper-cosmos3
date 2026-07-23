#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "quantize_a8.cuh"
#include "unpack_int4.cuh"

namespace {

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

extern "C" int piper_cutlass_w8a8_linear(
    const void* activation,
    const void* weight,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    void* output,
    int rows,
    int cols,
    int inner,
    cudaStream_t stream);

extern "C" int piper_cutlass_w4a8_fused_linear(
    const void* activation,
    const void* packed_weight,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    void* output,
    int rows,
    int cols,
    int inner,
    cudaStream_t stream);

torch::Tensor unpack_int4_cuda(torch::Tensor packed_qweight) {
    check_cuda_contiguous(packed_qweight, "packed_qweight");
    TORCH_CHECK(
        packed_qweight.scalar_type() == torch::kUInt8,
        "packed_qweight must be uint8");
    TORCH_CHECK(packed_qweight.dim() == 2, "packed_qweight must be 2D");

    const c10::cuda::CUDAGuard device_guard(packed_qweight.device());
    const int rows = static_cast<int>(packed_qweight.size(0));
    const int packed_cols = static_cast<int>(packed_qweight.size(1));
    auto output = torch::empty(
        {rows, packed_cols * 2},
        packed_qweight.options().dtype(torch::kInt8));
    const auto stream =
        at::cuda::getCurrentCUDAStream(packed_qweight.device().index());
    piper_unpack_int4(
        packed_qweight.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        rows,
        packed_cols,
        stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor w8a8_linear_cuda(
    torch::Tensor activation,
    torch::Tensor qweight,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> bias_optional) {
    check_cuda_contiguous(activation, "activation");
    check_cuda_contiguous(qweight, "qweight");
    check_cuda_contiguous(weight_scale, "weight_scale");
    TORCH_CHECK(
        activation.scalar_type() == torch::kBFloat16,
        "activation must be bfloat16");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt8, "qweight must be int8");
    TORCH_CHECK(
        weight_scale.scalar_type() == torch::kFloat32,
        "weight_scale must be float32");
    TORCH_CHECK(activation.dim() == 2, "activation must be [M,K]");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be [N,K]");
    TORCH_CHECK(weight_scale.dim() == 1, "weight_scale must be [N]");

    const int rows = static_cast<int>(activation.size(0));
    const int inner = static_cast<int>(activation.size(1));
    const int cols = static_cast<int>(qweight.size(0));
    TORCH_CHECK(qweight.size(1) == inner, "qweight K does not match activation");
    TORCH_CHECK(weight_scale.size(0) == cols, "weight_scale N mismatch");
    TORCH_CHECK(
        inner % 16 == 0,
        "CUTLASS INT8 kernel requires K divisible by 16, got ",
        inner);
    TORCH_CHECK(qweight.device() == activation.device(), "qweight device mismatch");
    TORCH_CHECK(
        weight_scale.device() == activation.device(),
        "weight_scale device mismatch");

    const c10::cuda::CUDAGuard device_guard(activation.device());
    const auto stream =
        at::cuda::getCurrentCUDAStream(activation.device().index());
    auto activation_q = torch::empty(
        {rows, inner},
        activation.options().dtype(torch::kInt8));
    auto activation_scale = torch::empty(
        {rows},
        activation.options().dtype(torch::kFloat32));
    auto output = torch::empty(
        {rows, cols},
        activation.options().dtype(torch::kBFloat16));

    piper_quantize_a8_rowwise(
        reinterpret_cast<const __nv_bfloat16*>(
            activation.data_ptr<at::BFloat16>()),
        activation_q.data_ptr<int8_t>(),
        activation_scale.data_ptr<float>(),
        rows,
        inner,
        stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const void* bias_pointer = nullptr;
    if (bias_optional.has_value() && bias_optional->defined()) {
        const auto& bias = *bias_optional;
        check_cuda_contiguous(bias, "bias");
        TORCH_CHECK(bias.device() == activation.device(), "bias device mismatch");
        TORCH_CHECK(
            bias.scalar_type() == torch::kBFloat16,
            "bias must be bfloat16");
        TORCH_CHECK(bias.dim() == 1 && bias.numel() == cols, "bias N mismatch");
        bias_pointer = bias.data_ptr<at::BFloat16>();
    }

    const int status = piper_cutlass_w8a8_linear(
        activation_q.data_ptr<int8_t>(),
        qweight.data_ptr<int8_t>(),
        activation_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        bias_pointer,
        output.data_ptr<at::BFloat16>(),
        rows,
        cols,
        inner,
        stream.stream());
    TORCH_CHECK(
        status == 0,
        "CUTLASS W8A8 GEMM failed with status ",
        status,
        " for M=",
        rows,
        ", N=",
        cols,
        ", K=",
        inner);
    return output;
}

torch::Tensor w4a8_linear_debug_cuda(
    torch::Tensor activation,
    torch::Tensor packed_qweight,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> bias_optional) {
    TORCH_CHECK(
        packed_qweight.dim() == 2,
        "packed_qweight must be [N,K/2]");
    TORCH_CHECK(
        activation.dim() == 2 &&
            packed_qweight.size(1) * 2 == activation.size(1),
        "packed_qweight K does not match activation");
    // Baseline implementation: unpack to a global INT8 temporary, then invoke
    // the real CUTLASS Tensor Core GEMM.  The next kernel milestone moves this
    // unpack into the CUTLASS mainloop.
    auto qweight = unpack_int4_cuda(packed_qweight);
    return w8a8_linear_cuda(
        activation,
        qweight,
        weight_scale,
        bias_optional);
}

torch::Tensor w4a8_linear_cuda(
    torch::Tensor activation,
    torch::Tensor packed_qweight,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> bias_optional) {
    check_cuda_contiguous(activation, "activation");
    check_cuda_contiguous(packed_qweight, "packed_qweight");
    check_cuda_contiguous(weight_scale, "weight_scale");
    TORCH_CHECK(
        activation.scalar_type() == torch::kBFloat16,
        "activation must be bfloat16");
    TORCH_CHECK(
        packed_qweight.scalar_type() == torch::kUInt8,
        "packed_qweight must be uint8");
    TORCH_CHECK(
        weight_scale.scalar_type() == torch::kFloat32,
        "weight_scale must be float32");
    TORCH_CHECK(activation.dim() == 2, "activation must be [M,K]");
    TORCH_CHECK(
        packed_qweight.dim() == 2,
        "packed_qweight must be [N,K/2]");
    TORCH_CHECK(weight_scale.dim() == 1, "weight_scale must be [N]");

    const int rows = static_cast<int>(activation.size(0));
    const int inner = static_cast<int>(activation.size(1));
    const int cols = static_cast<int>(packed_qweight.size(0));
    TORCH_CHECK(
        packed_qweight.size(1) * 2 == inner,
        "packed_qweight K does not match activation");
    TORCH_CHECK(weight_scale.size(0) == cols, "weight_scale N mismatch");
    TORCH_CHECK(
        inner % 16 == 0,
        "fused CUTLASS W4A8 kernel requires K divisible by 16, got ",
        inner);
    TORCH_CHECK(
        packed_qweight.device() == activation.device(),
        "packed_qweight device mismatch");
    TORCH_CHECK(
        weight_scale.device() == activation.device(),
        "weight_scale device mismatch");

    const c10::cuda::CUDAGuard device_guard(activation.device());
    const auto stream =
        at::cuda::getCurrentCUDAStream(activation.device().index());
    auto activation_q = torch::empty(
        {rows, inner},
        activation.options().dtype(torch::kInt8));
    auto activation_scale = torch::empty(
        {rows},
        activation.options().dtype(torch::kFloat32));
    auto output = torch::empty(
        {rows, cols},
        activation.options().dtype(torch::kBFloat16));

    piper_quantize_a8_rowwise(
        reinterpret_cast<const __nv_bfloat16*>(
            activation.data_ptr<at::BFloat16>()),
        activation_q.data_ptr<int8_t>(),
        activation_scale.data_ptr<float>(),
        rows,
        inner,
        stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const void* bias_pointer = nullptr;
    if (bias_optional.has_value() && bias_optional->defined()) {
        const auto& bias = *bias_optional;
        check_cuda_contiguous(bias, "bias");
        TORCH_CHECK(bias.device() == activation.device(), "bias device mismatch");
        TORCH_CHECK(
            bias.scalar_type() == torch::kBFloat16,
            "bias must be bfloat16");
        TORCH_CHECK(bias.dim() == 1 && bias.numel() == cols, "bias N mismatch");
        bias_pointer = bias.data_ptr<at::BFloat16>();
    }

    const int status = piper_cutlass_w4a8_fused_linear(
        activation_q.data_ptr<int8_t>(),
        packed_qweight.data_ptr<uint8_t>(),
        activation_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        bias_pointer,
        output.data_ptr<at::BFloat16>(),
        rows,
        cols,
        inner,
        stream.stream());
    TORCH_CHECK(
        status == 0,
        "fused CUTLASS W4A8 GEMM failed with status ",
        status,
        " for M=",
        rows,
        ", N=",
        cols,
        ", K=",
        inner);
    return output;
}

}  // namespace

TORCH_LIBRARY(piper_w4a8, library) {
    library.def("unpack_int4(Tensor packed_qweight) -> Tensor");
    library.def(
        "w8a8_linear(Tensor activation, Tensor qweight, "
        "Tensor weight_scale, Tensor? bias=None) -> Tensor");
    library.def(
        "linear(Tensor activation, Tensor packed_qweight, "
        "Tensor weight_scale, Tensor? bias=None) -> Tensor");
    library.def(
        "linear_debug(Tensor activation, Tensor packed_qweight, "
        "Tensor weight_scale, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(piper_w4a8, CUDA, library) {
    library.impl("unpack_int4", &unpack_int4_cuda);
    library.impl("w8a8_linear", &w8a8_linear_cuda);
    library.impl("linear", &w4a8_linear_cuda);
    library.impl("linear_debug", &w4a8_linear_debug_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("kernel_variant", []() {
        return "cutlass-w4a8-fused-mainloop";
    });
}
