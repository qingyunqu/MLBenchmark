
/*
  Generated by conv2d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "manifest.h"
#include "convolution/Conv2dOperation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2"
  using cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2_base = 
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
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      2,
      cutlass::half_t,
      cutlass::half_t
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
struct cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2 : 
  public cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2(Manifest &manifest) {


  using Operation_cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2>;

  manifest.append(new Conv2dOperation<
    Operation_cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2>(
      "cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align2"));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

