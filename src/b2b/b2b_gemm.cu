#include "Manifest.h"
#include "profile.h"
#include <cuda_fp16.h>

#include "b2b/GemmGemmOperation.h"
#include "b2b/device/b2b_gemm.h"
#include "matmul/GemmOperation.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = ElementAccumulator;

using ThreadblockShape0 = cutlass::gemm::GemmShape<256, 128, 32>;
using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
using ThreadblockShape1 = cutlass::gemm::GemmShape<256, 128, 32>;
using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

// clang-format off
using EpilogueOutputOp0 = 
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    InstructionShape::kM * InstructionShape::kN / 32,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
  >;

using EpilogueOutputOp1 = 
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    8, // 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
  >;

using Gemm0 = cutlass::gemm::device::Gemm<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm75,
  ThreadblockShape0,
  WarpShape0,
  InstructionShape,
  EpilogueOutputOp0,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  2,
  8,
  8,
  false,
  cutlass::arch::OpMultiplyAdd
>;

using Gemm1 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
  >;

using B2bGemm = cutlass::gemm::device::B2bGemm<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm75,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  2,
  8,
  8,
  false,
  cutlass::arch::OpMultiplyAdd
>;
// clang-format on

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
void initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8(
    Manifest &manifest);
} // namespace library
} // namespace cutlass

int main(int argc, char *argv[]) {
  Manifest manifest;
  manifest.append(new GemmGemmOperation<B2bGemm>("b2b_gemm"));
  manifest.append(new GemmOperation<Gemm0>("Gemm0"));
  manifest.append(new GemmOperation<Gemm1>("Gemm1"));
  cutlass::library::
      initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8(
          manifest);

  profile_gemm_gemm<__half, __half, __half, float>(
      manifest, 4096, 4096, 4096, LayoutEnum::RowMajor, LayoutEnum::ColumnMajor,
      LayoutEnum::RowMajor);

  return 0;
}
