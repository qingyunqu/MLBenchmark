#include "Manifest.h"
#include "profile.h"
#include "matmul/GemmOperation.h"

#include <cassert>
#include <cuda_fp16.h>
#include <string>
#include <unordered_set>

using Operation_cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_tn_align8 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::Default
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    1,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
    
  >;

int main() {
    Manifest manifest;
    manifest.append(new GemmOperation<Operation_cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_tn_align8>("test"));

    profile_gemm<__half, __half, __half, float>(manifest, 4095, 4096, 4096, LayoutEnum::RowMajor, LayoutEnum::ColumnMajor, LayoutEnum::RowMajor, EpilogueEnum::None, {});
}