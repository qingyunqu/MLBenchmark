
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


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2"
  using cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2_base = 
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    float, 
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationRelu<
      float,
      2,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    2,
    2
  >::Kernel;


// Derived class
struct cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2 : 
  public cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2(Manifest &manifest) {


  using Operation_cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2>;

  manifest.append(new Conv2dBiasOperation<
    Operation_cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2>(
      "cutlass_tensorop_s1688fprop_optimized_f16_256x128_32x2_nhwc_align2", EpilogueEnum::Relu));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

