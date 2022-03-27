#include "Manifest.h"
#include "profile.h"
#include <cuda_fp16.h>

#include "b2b/ConvConvOperation.h"
#include "b2b/device/b2b_implicit_gemm_convolution.h"

// clang-format off
using ElementA           = cutlass::half_t; 
using ElementB           = cutlass::half_t; 
using ElementC           = cutlass::half_t; 
using ElementAccumulator = cutlass::half_t; 
using ElementCompute     = cutlass::half_t; 

using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape0 = cutlass::gemm::GemmShape<32, 64, 32>;
using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 256, 32>;
using WarpShape1 = cutlass::gemm::GemmShape<32, 256, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using EpilogueOutputOp0 = 
  cutlass::epilogue::thread::LinearCombinationRelu<
    ElementC,
    InstructionShape::kM * InstructionShape::kN / 32,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
  >;

using EpilogueOutputOp1 = 
  cutlass::epilogue::thread::LinearCombinationRelu<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling
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

int main(int argc, char *argv[]) {
  Manifest manifest;
  manifest.append(new ConvConvOperation<B2bConv2dFprop>("b2b_conv2d"));

  int64_t N = 32;
  int64_t iH = 56;
  int64_t iW = 56;
  int64_t iC = 64;
  int64_t oC = 64;
  int64_t kH = 3;
  int64_t kW = 3;
  int64_t strideH = 1;
  int64_t strideW = 1;
  int64_t paddingH = 1;
  int64_t paddingW = 1;
  int64_t dilationH = 1;
  int64_t dilationW = 1;

  int64_t oH = (iH + 2 * paddingH - dilationH * (kH - 1) - 1) / strideH + 1;
  int64_t oW = (iW + 2 * paddingW - dilationW * (kW - 1) - 1) / strideW + 1;

  profile_conv2d<__half, __half, __half, float>(
    manifest, N, iH, iW, iC, oH, oW, oC, kH, kW, strideH, strideW, paddingH,
    paddingW, dilationH, dilationW);

  return 0;
}
