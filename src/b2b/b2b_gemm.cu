#include "Manifest.h"
#include "profile.h"
#include <cuda_fp16.h>

#include "b2b/GemmGemmOperation.h"
#include "b2b/device/b2b_gemm.h"

using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;

using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 64>;
using WarpShape0 = cutlass::gemm::GemmShape<32, 64, 64>;
using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape1 = cutlass::gemm::GemmShape<32, 128, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

// clang-format off
using EpilogueOutputOp0 = 
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    InstructionShape::kM * InstructionShape::kN / 32,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Nothing
  >;

using EpilogueOutputOp1 = 
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
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
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  2
>;
// clang-format on

int main(int argc, char *argv[]) {
  Manifest manifest;
  manifest.append(new GemmGemmOperation<B2bGemm>("b2b_gemm"));

  profile_gemm_gemm<__half, __half, __half, __half>(
      manifest, 2048, 2048, 2048, LayoutEnum::RowMajor, LayoutEnum::ColumnMajor,
      LayoutEnum::RowMajor);

  return 0;
}
