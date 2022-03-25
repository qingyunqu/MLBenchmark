#include "Operation.h"
#include "benchmark.h"
#include "check.h"
#include "convolution/CudnnConv.h"
#include "cutlass_dtype.h"
#include "manifest.h"
#include "matmul/CublasMatmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"

template <typename T>
__global__ void bias_add(T *bias, T *result, int64_t m, int64_t n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int column = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && column < m) {
    result[column * n + row] += bias[row];
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Gemm
/////////////////////////////////////////////////////////////////////////////////////////////////

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
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(TA)));
  CUDACHECK(cudaMalloc(&b, n * k * sizeof(TB)));
  CUDACHECK(cudaMalloc(&c, m * n * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_c, m * n * sizeof(TC)));
  RandCUDABuffer(a, m * k, -1.f, 1.f);
  RandCUDABuffer(b, n * k, -1.f, 1.f);
  RandCUDABuffer(c, m * n);
  FillCUDABuffer(ref_c, m * n);

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

  float cublas_time = benchmark<Op<TA, TC>>(op, stream, a, b, ref_c);

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
    kernel->SetArgument(m, n, k, (void *)a, (void *)b, (void *)c, (void *)c);
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream, nullptr);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(c, ref_c, m * n, 1e-1f);
    std::cout << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    float time = benchmark<Operation>(kernel, stream);
    std::cout << ", " << time << ", " << cublas_time << std::endl;
  }

  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
  delete op;
  CUBLASCHECK(cublasDestroy(handle));
  std::cout << "\n\n";
}

template void Manifest::profile_gemm<__half, __half, float, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile_gemm<__half, __half, __half, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile_gemm<__half, __half, __half, __half>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmBias
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void Manifest::profile_gemm_bias(int64_t m, int64_t n, int64_t k,
                                 LayoutEnum layout_a, LayoutEnum layout_b,
                                 LayoutEnum layout_c) {
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  TA *a = nullptr;
  TB *b = nullptr;
  TC *c = nullptr;
  TC *d = nullptr;
  TC *ref_d = nullptr;
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(TA)));
  CUDACHECK(cudaMalloc(&b, n * k * sizeof(TB)));
  assert(layout_c == LayoutEnum::RowMajor);
  CUDACHECK(cudaMalloc(&c, n * sizeof(TC))); // only RowMajor
  CUDACHECK(cudaMalloc(&d, m * n * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_d, m * n * sizeof(TC)));
  RandCUDABuffer(a, m * k, -1.f, 1.f);
  RandCUDABuffer(b, n * k, -1.f, 1.f);
  RandCUDABuffer(c, n, -1.f, 1.f); // only RowMajor
  RandCUDABuffer(d, m * n);
  FillCUDABuffer(ref_d, m * n);

  cudaStream_t stream = nullptr;
  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  Matmul<TA, TC> *op = new CublasMatmul<TA, TC, CompOn>(
      m, n, k, layout_a == LayoutEnum::ColumnMajor,
      layout_b == LayoutEnum::ColumnMajor, layout_c == LayoutEnum::ColumnMajor,
      handle);
  op->Run(a, b, ref_d);
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  bias_add<TC><<<grid, block, 0, stream>>>(c, ref_d, m, n);
  CUDACHECK(cudaDeviceSynchronize());

  typename Operation::OperationTrait trait{
      OperationEnum::MatmulBias,
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
    kernel->SetArgument(m, n, k, (void *)a, (void *)b, (void *)c, (void *)d);
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream, nullptr);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(d, ref_d, m * n, 1e-1f);
    std::cout << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    float time = benchmark<Operation>(kernel, stream);
    std::cout << ", " << time << std::endl;
  }

  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(d));
  CUDACHECK(cudaFree(ref_d));
  delete op;
  CUBLASCHECK(cublasDestroy(handle));
  std::cout << "\n\n";
}

template void Manifest::profile_gemm_bias<__half, __half, float, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile_gemm_bias<__half, __half, __half, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile_gemm_bias<__half, __half, __half, __half>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2d
/////////////////////////////////////////////////////////////////////////////////////////////////

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
  CUDACHECK(cudaMalloc(&input, N * iH * iW * iC * sizeof(TA)));
  CUDACHECK(cudaMalloc(&filter, oC * kH * kW * iC * sizeof(TB)));
  CUDACHECK(cudaMalloc(&output, N * oH * oW * oC * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_output, N * oH * oW * oC * sizeof(TC)));
  RandCUDABuffer(input, N * iH * iW * iC, -1.f, 1.f);
  RandCUDABuffer(filter, oC * kH * kW * iC, -1.f, 1.f);
  RandCUDABuffer(output, N * oH * oW * oC);
  FillCUDABuffer(ref_output, N * oH * oW * oC);

  cudaStream_t stream = nullptr;
  cudnnHandle_t handle;
  CUDNNCHECK(cudnnCreate(&handle));
  CUDNNCHECK(cudnnSetStream(handle, stream));
  Conv<TA, TC> *op = new CudnnConv<TA, TC, CompOn>(
      "NHWC", N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH,
      paddingW, dilationH, dilationW, handle);
  op->Run(input, filter, ref_output);
  CUDACHECK(cudaDeviceSynchronize());

  float cudnn_time =
      benchmark<Op<TA, TC>>(op, stream, input, filter, ref_output);

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
    std::cout << ", " << time << ", " << cudnn_time << std::endl;
  }

  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(filter));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFree(ref_output));
  delete op;
  CUDNNCHECK(cudnnDestroy(handle));
  std::cout << "\n\n";
}

template void Manifest::profile_conv2d<__half, __half, __half, float>(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template void Manifest::profile_conv2d<__half, __half, __half, __half>(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);