
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"

#include "matmul/GemmBiasOperation.h"
#include "Manifest.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_tensorop_s1688gemm_bias_f16_128x256_32x2_nt_align8
  using Operation_cutlass_tensorop_s1688gemm_bias_f16_128x256_32x2_nt_align8 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
    
  >;


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_s1688gemm_bias_f16_128x256_32x2_nt_align8(Manifest &manifest) {



  manifest.append(new GemmBiasOperation<Operation_cutlass_tensorop_s1688gemm_bias_f16_128x256_32x2_nt_align8>("cutlass_tensorop_s1688gemm_bias_f16_128x256_32x2_nt_align8", EpilogueEnum::None));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

