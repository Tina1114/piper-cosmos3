#include <cuda_runtime.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cute/tensor.hpp"

namespace piper {
namespace w4a8 {

using namespace cute;

using ElementA = int8_t;
using ElementB = int8_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
// A logical KxN column-major B has the same bytes as [N,K] row-major qweight.
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentOutput = 8;
using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int MainloopStages = 4;
constexpr int EpilogueStages = 1;

using OutputTileThreadMap =
    cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementOutput,
        AlignmentOutput,
        EpilogueStages>;

using AccumulatorFetch = cutlass::epilogue::threadblock::VisitorAccFetch;
using ActivationScaleLoad =
    cutlass::epilogue::threadblock::VisitorColBroadcast<
        OutputTileThreadMap,
        float,
        Stride<_1, _0, _0>>;
using WeightScaleLoad =
    cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap,
        float,
        Stride<_0, _1, int32_t>>;
using BiasLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap,
    ElementOutput,
    Stride<_0, _1, int32_t>>;
using MultiplyActivationScale =
    cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        float,
        float,
        cutlass::FloatRoundStyle::round_to_nearest>;
using MultiplyWeightScale =
    cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        float,
        float,
        cutlass::FloatRoundStyle::round_to_nearest>;
using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus,
    float,
    float,
    cutlass::FloatRoundStyle::round_to_nearest>;
using StoreOutput = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap,
    ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

using AccTimesActivationScale =
    cutlass::epilogue::threadblock::Sm80EVT<
        MultiplyActivationScale,
        AccumulatorFetch,
        ActivationScaleLoad>;
using AccTimesBothScales = cutlass::epilogue::threadblock::Sm80EVT<
    MultiplyWeightScale,
    AccTimesActivationScale,
    WeightScaleLoad>;
using NoBiasEpilogue = cutlass::epilogue::threadblock::Sm80EVT<
    StoreOutput,
    AccTimesBothScales>;
using AddBiasTree = cutlass::epilogue::threadblock::Sm80EVT<
    AddBias,
    AccTimesBothScales,
    BiasLoad>;
using BiasEpilogue = cutlass::epilogue::threadblock::Sm80EVT<
    StoreOutput,
    AddBiasTree>;

template <class EpilogueVisitor>
using GemmKernelFor =
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        AlignmentA,
        ElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        AlignmentB,
        ElementOutput,
        LayoutOutput,
        AlignmentOutput,
        ElementAccumulator,
        ElementCompute,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueVisitor,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        MainloopStages,
        cutlass::arch::OpMultiplyAddSaturate,
        EpilogueStages>::GemmKernel;

using NoBiasKernel = GemmKernelFor<NoBiasEpilogue>;
using NoBiasGemm =
    cutlass::gemm::device::GemmUniversalAdapter<NoBiasKernel>;
using BiasKernel = GemmKernelFor<BiasEpilogue>;
using BiasGemm = cutlass::gemm::device::GemmUniversalAdapter<BiasKernel>;

static typename StoreOutput::Arguments output_arguments(
    void* output,
    int rows,
    int cols) {
    return {
        reinterpret_cast<ElementOutput*>(output),
        {
            static_cast<int64_t>(cols),
            _1{},
            static_cast<int64_t>(rows) * cols,
        },
    };
}

static int run_no_bias(
    const void* activation,
    const void* weight,
    const void* activation_scale,
    const void* weight_scale,
    void* output,
    int rows,
    int cols,
    int inner,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(rows, cols, inner);
    typename NoBiasEpilogue::Arguments epilogue{
        {
            {
                {},
                {
                    reinterpret_cast<const float*>(activation_scale),
                    1.0f,
                    {},
                },
                {},
            },
            {
                reinterpret_cast<const float*>(weight_scale),
                1.0f,
                {_0{}, _1{}, int32_t(cols)},
            },
            {},
        },
        output_arguments(output, rows, cols),
    };
    typename NoBiasGemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        1,
        epilogue,
        reinterpret_cast<const ElementA*>(activation),
        reinterpret_cast<const ElementB*>(weight),
        nullptr,
        nullptr,
        static_cast<int64_t>(rows) * inner,
        static_cast<int64_t>(cols) * inner,
        0,
        0,
        inner,
        inner,
        cols,
        cols);

    NoBiasGemm gemm;
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return static_cast<int>(status) | 0x10000;
    }
    if (NoBiasGemm::get_workspace_size(arguments) != 0) {
        return 0x40000;
    }
    status = gemm.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return static_cast<int>(status) | 0x20000;
    }
    status = gemm.run(stream);
    return status == cutlass::Status::kSuccess
        ? 0
        : (static_cast<int>(status) | 0x30000);
}

static int run_bias(
    const void* activation,
    const void* weight,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    void* output,
    int rows,
    int cols,
    int inner,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(rows, cols, inner);
    typename BiasEpilogue::Arguments epilogue{
        {
            {
                {
                    {},
                    {
                        reinterpret_cast<const float*>(activation_scale),
                        1.0f,
                        {},
                    },
                    {},
                },
                {
                    reinterpret_cast<const float*>(weight_scale),
                    1.0f,
                    {_0{}, _1{}, int32_t(cols)},
                },
                {},
            },
            {
                reinterpret_cast<const ElementOutput*>(bias),
                ElementOutput(0),
                {_0{}, _1{}, int32_t(cols)},
            },
            {},
        },
        output_arguments(output, rows, cols),
    };
    typename BiasGemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        1,
        epilogue,
        reinterpret_cast<const ElementA*>(activation),
        reinterpret_cast<const ElementB*>(weight),
        nullptr,
        nullptr,
        static_cast<int64_t>(rows) * inner,
        static_cast<int64_t>(cols) * inner,
        0,
        0,
        inner,
        inner,
        cols,
        cols);

    BiasGemm gemm;
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return static_cast<int>(status) | 0x10000;
    }
    if (BiasGemm::get_workspace_size(arguments) != 0) {
        return 0x40000;
    }
    status = gemm.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return static_cast<int>(status) | 0x20000;
    }
    status = gemm.run(stream);
    return status == cutlass::Status::kSuccess
        ? 0
        : (static_cast<int>(status) | 0x30000);
}

}  // namespace w4a8
}  // namespace piper

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
    cudaStream_t stream) {
    if (bias != nullptr) {
        return piper::w4a8::run_bias(
            activation,
            weight,
            activation_scale,
            weight_scale,
            bias,
            output,
            rows,
            cols,
            inner,
            stream);
    }
    return piper::w4a8::run_no_bias(
        activation,
        weight,
        activation_scale,
        weight_scale,
        output,
        rows,
        cols,
        inner,
        stream);
}
