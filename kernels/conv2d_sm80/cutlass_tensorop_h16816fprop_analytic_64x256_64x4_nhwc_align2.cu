
/*
  Generated by conv2d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "Manifest.h"
#include "convolution/Conv2dOperation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Analytic kernel instance "cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2"
  using cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2_base = 
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64 >,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      2,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::Default
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided,
    2,
    2
  >::Kernel;


// Derived class
struct cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2 : 
  public cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2(Manifest &manifest) {


  using Operation_cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2>;

  manifest.append(new Conv2dOperation<
    Operation_cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2>(
      "cutlass_tensorop_h16816fprop_analytic_64x256_64x4_nhwc_align2", EpilogueEnum::None));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

