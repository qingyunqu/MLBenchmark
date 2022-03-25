
/*
  Generated by conv2d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "Manifest.h"
#include "convolution/Conv2dBiasOperation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop_bias Analytic kernel instance "cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8"
  using cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8_base = 
  typename cutlass::conv::kernel::DefaultConv2dFprop_bias<
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32 >,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
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
struct cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8 : 
  public cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8(Manifest &manifest) {


  using Operation_cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8>;

  manifest.append(new Conv2dBiasOperation<
    Operation_cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8>(
      "cutlass_tensorop_f16_s884fprop_bias_analytic_f16_64x64_32x2_nhwc_align8", EpilogueEnum::None));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

