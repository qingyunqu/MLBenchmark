
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


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4"
  using cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4_base = 
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64 >,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      4,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::Default
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;


// Derived class
struct cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4 : 
  public cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4(Manifest &manifest) {


  using Operation_cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4>;

  manifest.append(new Conv2dOperation<
    Operation_cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4>(
      "cutlass_tensorop_f16_s16816fprop_optimized_f16_64x256_64x4_nhwc_align4", EpilogueEnum::None));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

