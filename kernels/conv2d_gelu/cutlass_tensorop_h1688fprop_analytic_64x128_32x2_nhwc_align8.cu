
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


  // Conv2dFprop Analytic kernel instance "cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8"
  using cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8_base = 
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationGELU<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::Default
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8
  >::Kernel;


// Derived class
struct cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8 : 
  public cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8(Manifest &manifest) {


  using Operation_cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8>;

  manifest.append(new Conv2dOperation<
    Operation_cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8>(
      "cutlass_tensorop_h1688fprop_analytic_64x128_32x2_nhwc_align8", EpilogueEnum::Grelu));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

