
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


  // Conv2dFprop Analytic kernel instance "cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1"
  using cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1_base = 
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32 >,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombinationRelu<
      cutlass::half_t,
      1,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided,
    1,
    1
  >::Kernel;


// Derived class
struct cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1 : 
  public cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1(Manifest &manifest) {


  using Operation_cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1>;

  manifest.append(new Conv2dBiasOperation<
    Operation_cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1>(
      "cutlass_tensorop_h884fprop_analytic_64x64_32x2_nhwc_align1", EpilogueEnum::Relu));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

