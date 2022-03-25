
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


  // Conv2dFprop Analytic kernel instance "cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4"
  using cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4_base = 
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    float, 
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32 >,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;


// Derived class
struct cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4 : 
  public cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4(Manifest &manifest) {


  using Operation_cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4>;

  manifest.append(new Conv2dOperation<
    Operation_cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4>(
      "cutlass_tensorop_s884fprop_analytic_f16_64x128_32x2_nhwc_align4"));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

