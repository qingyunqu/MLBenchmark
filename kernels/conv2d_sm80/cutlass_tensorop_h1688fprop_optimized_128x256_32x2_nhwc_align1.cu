
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


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1"
  using cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1_base = 
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
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      1,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::Default
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    1,
    1
  >::Kernel;


// Derived class
struct cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1 : 
  public cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1(Manifest &manifest) {


  using Operation_cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1>;

  manifest.append(new Conv2dOperation<
    Operation_cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1>(
      "cutlass_tensorop_h1688fprop_optimized_128x256_32x2_nhwc_align1", EpilogueEnum::None));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

