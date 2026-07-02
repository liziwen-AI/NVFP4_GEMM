import pathlib

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


# ---- C++ stub: declare the function so load_inline can bind it ----
gemm_cpp = r"""
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_gemm(torch::Tensor A,
                              torch::Tensor B,
                              torch::Tensor SFA,
                              torch::Tensor SFB,
                              torch::Tensor C,
                              int64_t kernel_type);
"""

# ---- CUDA source: CUTLASS-based blockscaled GEMM with multiple kernels ----
gemm_cuda = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/cutlass.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

namespace {

// Common types for all kernels
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// ============================================================================
// Kernel 0: 1SM kernel (128, 128, 256) with cluster (1,1,1)
// Best for large K problems
// ============================================================================
namespace kernel_1sm {

using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace kernel_1sm

// ============================================================================
// Kernel 1: 2SM kernel (256, 128, 256) with cluster (2,1,1)
// Better for medium K problems
// ============================================================================
namespace kernel_2sm {

using MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_256>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace kernel_2sm

// ============================================================================
// Kernel 2: 1SM kernel (128, 128, 256) with cluster (1,2,1) for N multicast
// For wide N problems
// ============================================================================
namespace kernel_1sm_n_multicast {

using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace kernel_1sm_n_multicast

// Template runner for any GEMM type
template<typename GemmType>
torch::Tensor run_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor SFA,
                        torch::Tensor SFB, torch::Tensor C,
                        int m, int n, int logical_k, int batch) {
    using Gemm = GemmType;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using Sm1xxBlkScaledConfig =
        typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, logical_k, batch});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, logical_k, batch});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, batch});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, batch});

    LayoutSFA layout_SFA =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, logical_k, batch));
    LayoutSFB layout_SFB =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, logical_k, batch));

    auto args = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, logical_k, batch},
        {
            reinterpret_cast<ElementA::DataType const*>(A.data_ptr()),
            stride_A,
            reinterpret_cast<ElementB::DataType const*>(B.data_ptr()),
            stride_B,
            reinterpret_cast<ElementA::ScaleFactorType const*>(SFA.data_ptr()),
            layout_SFA,
            reinterpret_cast<ElementB::ScaleFactorType const*>(SFB.data_ptr()),
            layout_SFB,
        },
        {
            {1.0f, 0.0f},
            reinterpret_cast<ElementC const*>(C.data_ptr()),
            stride_C,
            reinterpret_cast<ElementD*>(C.data_ptr()),
            stride_D,
        }};

    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty(
        {static_cast<long long>(workspace_size)},
        torch::dtype(torch::kUInt8).device(C.device()));

    Gemm gemm_op;
    auto status = gemm_op.initialize(
        args,
        workspace_size ? workspace.data_ptr() : nullptr);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM init failed");

    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM run failed");

    return C;
}

}  // namespace

torch::Tensor cuda_nvfp4_gemm(torch::Tensor A,
                              torch::Tensor B,
                              torch::Tensor SFA,
                              torch::Tensor SFB,
                              torch::Tensor C,
                              int64_t kernel_type) {
    c10::cuda::CUDAGuard device_guard(A.device());
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(SFA.is_cuda(), "SFA must be CUDA tensor");
    TORCH_CHECK(SFB.is_cuda(), "SFB must be CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA tensor");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && C.dim() == 3,
                "A, B, C must be rank-3 tensors");

    const int m = static_cast<int>(A.size(0));
    const int k_packed = static_cast<int>(A.size(1));
    const int batch = static_cast<int>(A.size(2));
    const int n = static_cast<int>(B.size(0));
    const int logical_k = k_packed * 2;  // packed fp4_x2 -> logical K

    switch (kernel_type) {
        case 0:  // 1SM kernel (128,128,256) cluster (1,1,1)
            return run_gemm<kernel_1sm::Gemm>(A, B, SFA, SFB, C, m, n, logical_k, batch);
        case 1:  // 2SM kernel (256,128,256) cluster (2,1,1)
            return run_gemm<kernel_2sm::Gemm>(A, B, SFA, SFB, C, m, n, logical_k, batch);
        case 2:  // 1SM with N multicast (128,128,256) cluster (1,2,1)
            return run_gemm<kernel_1sm_n_multicast::Gemm>(A, B, SFA, SFB, C, m, n, logical_k, batch);
        default:
            TORCH_CHECK(false, "Invalid kernel_type: ", kernel_type);
    }
}
"""

# ---- build the extension ----
repo_root = pathlib.Path(__file__).resolve().parent
include_paths = [
    str(repo_root / "cutlass" / "include"),
    str(repo_root / "cutlass" / "tools" / "util" / "include"),
]

nvfp4_gemm_module = load_inline(
    name="nvfp4_gemm_cutlass",
    cpp_sources=[gemm_cpp],
    cuda_sources=[gemm_cuda],
    functions=["cuda_nvfp4_gemm"],
    extra_include_paths=include_paths,
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3",
        "-w",
        "-maxrregcount=128",
        "--use_fast_math",
        "-allow-unsupported-compiler",
    ],
    extra_ldflags=["-lcuda", "-lcublas"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    CUDA port of best_kernel_gemm.py using a CUTLASS blockscaled GEMM.
    Uses size-based kernel selection for optimal performance.

    Args:
        data: (a, b, sfa_cpu, sfb_cpu, sfa_permuted, sfb_permuted, c)

    Kernel types:
        0: 1SM (128,128,256) cluster (1,1,1) - for large K
        1: 2SM (256,128,256) cluster (2,1,1) - for medium K, M-multicast
        2: 1SM (128,128,256) cluster (1,2,1) - for small K, N-multicast
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    # Get problem dimensions
    k = a.size(1) * 2  # packed fp4_x2 -> logical K

    # Select kernel based on K size
    # Benchmarks show:
    # - Large K (>=8192): 1SM kernel works best
    # - Medium K (4096-8191): 2SM with M-multicast is fastest
    # - Small K (<4096): 1SM with N-multicast helps
    if k >= 8192:
        kernel_type = 0  # 1SM kernel
    elif k >= 4096:
        kernel_type = 1  # 2SM with M-multicast
    else:
        kernel_type = 2  # 1SM with N-multicast

    return nvfp4_gemm_module.cuda_nvfp4_gemm(a, b, sfa_permuted, sfb_permuted, c, kernel_type)
