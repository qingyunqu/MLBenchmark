
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


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8"
  using cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8_base = 
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
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::Default
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8
  >::Kernel;


// Derived class
struct cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8 : 
  public cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8(Manifest &manifest) {


  using Operation_cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8>;

  manifest.append(new Conv2dOperation<
    Operation_cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8>(
      "cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x4_nhwc_align8", EpilogueEnum::None));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

