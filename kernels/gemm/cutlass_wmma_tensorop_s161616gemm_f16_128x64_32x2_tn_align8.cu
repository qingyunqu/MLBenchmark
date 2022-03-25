
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"

#include "matmul/GemmOperation.h"
#include "manifest.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_wmma_tensorop_s161616gemm_f16_128x64_32x2_tn_align8
  using Operation_cutlass_wmma_tensorop_s161616gemm_f16_128x64_32x2_tn_align8 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassWmmaTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::Default
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

void initialize_cutlass_wmma_tensorop_s161616gemm_f16_128x64_32x2_tn_align8(Manifest &manifest) {


#if defined(CUTLASS_ARCH_WMMA_SM70_ENABLED)
  manifest.append(new GemmOperation<Operation_cutlass_wmma_tensorop_s161616gemm_f16_128x64_32x2_tn_align8>("cutlass_wmma_tensorop_s161616gemm_f16_128x64_32x2_tn_align8"));
#endif


}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

