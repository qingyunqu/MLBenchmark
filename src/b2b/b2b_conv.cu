#include "Manifest.h"
#include "convolution/Conv2dOperation.h"
#include "profile.h"
#include <cuda_fp16.h>

#include "b2b/ConvConvOperation.h"
#include "b2b/device/b2b_implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"

// clang-format off
using ElementA           = cutlass::half_t; 
using ElementB           = cutlass::half_t; 
using ElementC           = cutlass::half_t; 
using ElementAccumulator = float; 
using ElementCompute     = ElementAccumulator; 

using ThreadblockShape0 = cutlass::gemm::GemmShape<256, 128, 32>;
using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using Conv2dFpropKernel0 = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementA, cutlass::layout::TensorNHWC,
  ElementB, cutlass::layout::TensorNHWC,
  ElementC, cutlass::layout::TensorNHWC,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm75,
  ThreadblockShape0,
  WarpShape0,
  InstructionShape,
  cutlass::epilogue::thread::LinearCombinationRelu<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementCompute//,
    //cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  2,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using Conv2dFprop0 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel0>;

using Conv2dFpropKernel1 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using Conv2dFprop1 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel1>;


using EpilogueOutputOp0 = 
  cutlass::epilogue::thread::LinearCombinationRelu<
    ElementC,
    InstructionShape::kM * InstructionShape::kN / 32,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Nothing
  >;

using EpilogueOutputOp1 = 
  cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Nothing
  >;

const bool SmemAccumulator = false;
using B2bConv2dFpropKernel = typename cutlass::conv::kernel::DefaultB2bConv2dFprop<
  ElementA, cutlass::layout::TensorNHWC,
  ElementB, cutlass::layout::TensorNHWC,
  ElementC, cutlass::layout::TensorNHWC,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm75,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  2,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  SmemAccumulator
>::Kernel;

using B2bConv2dFprop = cutlass::conv::device::B2bImplicitGemmConvolution<B2bConv2dFpropKernel>;
//clang-format on

// cutlass::conv::Conv2dProblemSize conv2d_f16_sm75_problem_size_0 (
//   {128, 56, 56, 64},    // input size (NHWC)
//   {64, 3, 3, 64},   // filter size (KRSC)
//   {1, 1, 1, 1},     // padding (pad_h, _, pad_w, _)
//   {1, 1},           // stride (stride_h, stride_w)
//   {1, 1},           // dilation (dilation_h, dilation_w)
//   {128, 56, 56, 64}     // output size (NPQK)
// );
// cutlass::conv::Conv2dProblemSize conv2d_f16_sm75_problem_size_1 (
//   {128, 56, 56, 64},    // input size (NHWC)
//   {256, 1, 1, 64},   // filter size (KRSC)
//   {0, 0, 0, 0},     // padding (pad_h, _, pad_w, _)
//   {1, 1},           // stride (stride_h, stride_w)
//   {1, 1},           // dilation (dilation_h, dilation_w)
//   {128, 56, 56, 256}     // output size (NPQK)
// );

int main(int argc, char *argv[]) {
  Manifest manifest;
  manifest.append(new Conv2dOperation<Conv2dFprop0>("Conv2d0"));
  manifest.append(new Conv2dOperation<Conv2dFprop1>("Conv2d1"));
  manifest.append(new ConvConvOperation<B2bConv2dFprop>("b2b_conv2d"));

  int64_t N0 = 128, iH0 = 56, iW0 = 56, iC0 = 64;
  int64_t oC0 = 64;
  int64_t kH0 = 3;
  int64_t kW0 = 3;
  int64_t strideH0 = 1, strideW0 = 1;
  int64_t paddingH0 = 1, paddingW0 = 1;
  int64_t dilationH0 = 1, dilationW0 = 1;

  int64_t oH0 = (iH0 + 2 * paddingH0 - dilationH0 * (kH0 - 1) - 1) / strideH0 + 1;
  int64_t oW0 = (iW0 + 2 * paddingW0 - dilationW0 * (kW0 - 1) - 1) / strideW0 + 1;
  printf("oH0: %d, oW0: %d\n", oH0, oW0);


  int64_t N1 = N0, iH1 = oH0, iW1 = oW0, iC1 = oC0;
  int64_t oC1 = 256;
  int64_t kH1 = 1;
  int64_t kW1 = 1;
  int64_t strideH1 = 1, strideW1 = 1;
  int64_t paddingH1 = 0, paddingW1 = 0;
  int64_t dilationH1 = 1, dilationW1 = 1;

  int64_t oH1 = (iH1 + 2 * paddingH1 - dilationH1 * (kH1 - 1) - 1) / strideH1 + 1;
  int64_t oW1 = (iW1 + 2 * paddingW1 - dilationW1 * (kW1 - 1) - 1) / strideW1 + 1;
  printf("oH1: %d, oW1: %d\n", oH1, oW1);


  profile_conv2d_conv2d<__half, __half, __half, float>(
    manifest, N0, iH0, iW0, iC0, oH0, oW0, oC0, kH0, kW0, strideH0, strideW0, paddingH0,
    paddingW0, dilationH0, dilationW0, N1, iH1, iW1, iC1, oH1, oW1, oC1, kH1, kW1, strideH1, strideW1, paddingH1,
    paddingW1, dilationH1, dilationW1);

  return 0;
}
