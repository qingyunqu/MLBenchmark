
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"

#include "matmul/GemmOperation.h"
#include "Manifest.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_tensorop_h884gemm_256x128_32x2_tn_align4
  using Operation_cutlass_tensorop_h884gemm_256x128_32x2_tn_align4 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      4,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::Default
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    4,
    4,
    false,
    cutlass::arch::OpMultiplyAdd
    
  >;


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_h884gemm_256x128_32x2_tn_align4(Manifest &manifest) {



  manifest.append(new GemmOperation<Operation_cutlass_tensorop_h884gemm_256x128_32x2_tn_align4>("cutlass_tensorop_h884gemm_256x128_32x2_tn_align4"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

