// Fused packed-W4 -> INT8 shared-memory mainloop for SM80/SM89.
//
// The global B iterator reads cutlass::int4b_t directly from the packed
// checkpoint layout. MmaPipelined's TransformB converts each register fragment
// to int8_t before the shared-memory store. The warp MMA therefore remains the
// mature s8*s8->s32 Tensor Core path, without a global unpacked-weight tensor.

#include <cuda_runtime.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cute/tensor.hpp"

namespace piper {
namespace w4a8_fused {

using namespace cute;

using ElementA = int8_t;
using ElementBGlobal = cutlass::int4b_t;
using ElementBMma = int8_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

constexpr int AlignmentA = 16;
// 16 logical INT4 values = one 64-bit global access.
constexpr int AlignmentBGlobal = 16;
constexpr int AlignmentBMma = 16;
constexpr int AlignmentOutput = 8;
using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int MainloopStages = 2;
constexpr int EpilogueStages = 1;
using Operator = cutlass::arch::OpMultiplyAddSaturate;
using Swizzle =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Build the MMA core for s8*s8 Tensor Core instructions and INT8 shared memory.
using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    ElementA,
    LayoutA,
    ElementBMma,
    LayoutB,
    ElementAccumulator,
    LayoutOutput,
    OperatorClass,
    MainloopStages,
    Operator>;

using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
    cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
    ElementA,
    LayoutA,
    1,
    typename MmaCore::IteratorThreadMapA,
    AlignmentA>;

// Crucial mixed-storage iterator: logical B is KxN column-major INT4. Its
// physical bytes are identical to the checkpoint's UINT8 [N,K/2] tensor.
using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
    cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
    ElementBGlobal,
    LayoutB,
    0,
    typename MmaCore::IteratorThreadMapB,
    AlignmentBGlobal>;

using TransformA = cutlass::NumericArrayConverter<
    typename MmaCore::SmemIteratorA::Element,
    typename IteratorA::Element,
    IteratorA::Fragment::kElements>;
using TransformB = cutlass::NumericArrayConverter<
    typename MmaCore::SmemIteratorB::Element,
    typename IteratorB::Element,
    IteratorB::Fragment::kElements>;

using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
    typename MmaCore::Shape,
    IteratorA,
    typename MmaCore::SmemIteratorA,
    IteratorB,
    typename MmaCore::SmemIteratorB,
    ElementAccumulator,
    LayoutOutput,
    typename MmaCore::MmaPolicy,
    TransformA,
    TransformB>;

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
using NoBiasCallbacks = cutlass::epilogue::threadblock::Sm80EVT<
    StoreOutput,
    AccTimesBothScales>;
using AddBiasTree = cutlass::epilogue::threadblock::Sm80EVT<
    AddBias,
    AccTimesBothScales,
    BiasLoad>;
using BiasCallbacks = cutlass::epilogue::threadblock::Sm80EVT<
    StoreOutput,
    AddBiasTree>;

// Reuse CUTLASS's validated EVT epilogue construction. Only its Mma type is
// replaced by the mixed global-W4/shared-W8 mainloop above.
template <class Callbacks>
using EpilogueConfig = cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    AlignmentA,
    ElementBMma,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    AlignmentBMma,
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
    Callbacks,
    Swizzle,
    MainloopStages,
    Operator,
    EpilogueStages>;

template <class Callbacks>
using KernelFor = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
    ThreadblockMma,
    typename EpilogueConfig<Callbacks>::Epilogue,
    Swizzle>;

using NoBiasKernel = KernelFor<NoBiasCallbacks>;
using NoBiasGemm =
    cutlass::gemm::device::GemmUniversalAdapter<NoBiasKernel>;
using BiasKernel = KernelFor<BiasCallbacks>;
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
    const void* packed_weight,
    const void* activation_scale,
    const void* weight_scale,
    void* output,
    int rows,
    int cols,
    int inner,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(rows, cols, inner);
    typename NoBiasCallbacks::Arguments epilogue{
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
        reinterpret_cast<const ElementBGlobal*>(packed_weight),
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
    const void* packed_weight,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    void* output,
    int rows,
    int cols,
    int inner,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(rows, cols, inner);
    typename BiasCallbacks::Arguments epilogue{
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
        reinterpret_cast<const ElementBGlobal*>(packed_weight),
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

}  // namespace w4a8_fused
}  // namespace piper

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
    cudaStream_t stream) {
    if (bias != nullptr) {
        return piper::w4a8_fused::run_bias(
            activation,
            packed_weight,
            activation_scale,
            weight_scale,
            bias,
            output,
            rows,
            cols,
            inner,
            stream);
    }
    return piper::w4a8_fused::run_no_bias(
        activation,
        packed_weight,
        activation_scale,
        weight_scale,
        output,
        rows,
        cols,
        inner,
        stream);
}
