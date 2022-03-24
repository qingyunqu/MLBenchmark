#include "manifest.h"
#include "Operation.h"
#include "benchmark.h"
#include "check.h"
#include "convolution/CudnnConv.h"
#include "convolution/util.h"
#include "cutlass_dtype.h"
#include "matmul/CublasMatmul.h"
#include "matmul/util.h"

#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"

template <typename TA, typename TB, typename TC, typename CompOn>
void Manifest::profile_gemm(int64_t m, int64_t n, int64_t k,
                            LayoutEnum layout_a, LayoutEnum layout_b,
                            LayoutEnum layout_c) {
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  TA *a = nullptr;
  TB *b = nullptr;
  TC *c = nullptr;
  TC *ref_c = nullptr;
  InitMatmulTensor(m, n, k, a, b, c, ref_c);

  cudaStream_t stream = nullptr;
  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  Matmul<TA, TC> *op = new CublasMatmul<TA, TC, CompOn>(
      m, n, k, layout_a == LayoutEnum::ColumnMajor,
      layout_b == LayoutEnum::ColumnMajor, layout_c == LayoutEnum::ColumnMajor,
      handle);
  op->Run(a, b, ref_c);
  CUDACHECK(cudaDeviceSynchronize());

  typename Operation::OperationTrait trait{
      OperationEnum::Matmul,
      cutlass_type_to_dtype_v<ElementInputA>,
      layout_a,
      cutlass_type_to_dtype_v<ElementInputB>,
      layout_b,
      cutlass_type_to_dtype_v<ElementOutput>,
      layout_c,
      cutlass_type_to_dtype_v<ElementAccumulator>};
  for (auto &kernel : kernels) {
    if (kernel->Trait() != trait) {
      continue;
    }
    kernel->SetArgument(m, n, k, (void *)a, (void *)b, (void *)c);
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream, nullptr);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(c, ref_c, m * n, 1e-1f);
    std::cout << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    float time = benchmark<Operation>(kernel, stream);
    std::cout << ", " << time << std::endl;
  }
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
  delete op;
  CUBLASCHECK(cublasDestroy(handle));
}

template void Manifest::profile_gemm<__half, __half, float, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile_gemm<__half, __half, __half, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile_gemm<__half, __half, __half, __half>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);

template <typename TA, typename TB, typename TC, typename CompOn>
void Manifest::profile_conv2d(int64_t N, int64_t iH, int64_t iW, int64_t iC,
                              int64_t oH, int64_t oW, int64_t oC, int64_t kH,
                              int64_t kW, int64_t strideH, int64_t strideW,
                              int64_t paddingH, int64_t paddingW,
                              int64_t dilationH /* = 1*/,
                              int64_t dilationW /* = 1*/) {
  // only profile NHWC layout
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  TA *input = nullptr;
  TB *filter = nullptr;
  TC *output = nullptr;
  TC *ref_output = nullptr;
  InitConv2dTensor(N, iW, iW, iC, oH, oW, oC, kH, kW, input, filter, output,
                   ref_output);

  cudaStream_t stream = nullptr;
  cudnnHandle_t handle;
  CUDNNCHECK(cudnnCreate(&handle));
  CUDNNCHECK(cudnnSetStream(handle, stream));
  Conv<TA, TC> *op = new CudnnConv<TA, TC, CompOn>(
      "NHWC", N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH,
      paddingW, dilationH, dilationW, handle);
  op->Run(input, output, ref_output);
  CUDACHECK(cudaDeviceSynchronize());

  typename Operation::OperationTrait trait{
      OperationEnum::Conv2d, cutlass_type_to_dtype_v<ElementInputA>,
      LayoutEnum::NHWC,      cutlass_type_to_dtype_v<ElementInputB>,
      LayoutEnum::NHWC,      cutlass_type_to_dtype_v<ElementOutput>,
      LayoutEnum::NHWC,      cutlass_type_to_dtype_v<ElementAccumulator>};
  for (auto kernel : kernels) {
    if (kernel->Trait() != trait) {
      continue;
    }
    kernel->SetArgument(N, iH, iW, iC, oH, oW, oC, kH, kW, strideH, strideW,
                        paddingH, paddingW, dilationH, dilationW, (void *)input,
                        (void *)filter, (void *)output);
    if (!kernel->Check()) {
      continue;
    }
    cutlass::device_memory::allocation<uint8_t> workspace(
        kernel->GetWorkspaceSize());
    kernel->Initialize(stream, workspace.get());
    kernel->Run();
    bool passed =
        CheckCUDABuffer<TC>(output, ref_output, N * oH * oW * oC, 1e-1f);
    std::cout << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    float time = benchmark<Operation>(kernel, stream);
    std::cout << ", " << time << std::endl;
  }

  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(filter));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFree(ref_output));
  delete op;
  CUDNNCHECK(cudnnDestroy(handle));
}

template void Manifest::profile_conv2d<__half, __half, __half, float>(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template void Manifest::profile_conv2d<__half, __half, __half, __half>(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);