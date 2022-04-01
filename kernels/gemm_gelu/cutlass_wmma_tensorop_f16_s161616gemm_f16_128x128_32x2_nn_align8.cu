
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"

#include "matmul/GemmOperation.h"
#include "Manifest.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_wmma_tensorop_f16_s161616gemm_f16_128x128_32x2_nn_align8
  using Operation_cutlass_wmma_tensorop_f16_s161616gemm_f16_128x128_32x2_nn_align8 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassWmmaTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
      cutlass::half_t,
      8,
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

void initialize_cutlass_wmma_tensorop_f16_s161616gemm_f16_128x128_32x2_nn_align8(Manifest &manifest) {


#if defined(CUTLASS_ARCH_WMMA_SM70_ENABLED)
  manifest.append(new GemmOperation<Operation_cutlass_wmma_tensorop_f16_s161616gemm_f16_128x128_32x2_nn_align8>("cutlass_wmma_tensorop_f16_s161616gemm_f16_128x128_32x2_nn_align8", EpilogueEnum::Grelu));
#endif


}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

