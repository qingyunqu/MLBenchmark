#include "Manifest.h"
#include "Operation.h"
#include "benchmark.h"
#include "check.h"
#include "convolution/CudnnConv.h"
#include "cutlass_dtype.h"
#include "matmul/CublasMatmul.h"
#include "profile.h"
#include "util/kernel.cuh"
#include "util/util.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"

#define DEBUG 1

struct Result {
  const char *kernel_name;
  bool passed;
  float cutlass_time;
  float library_time;
  bool operator<(const Result &result) const {
    return cutlass_time < result.cutlass_time;
  }
};

LayoutEnum transpose_matrix(LayoutEnum a) {
  if (a == LayoutEnum::RowMajor) {
    return LayoutEnum::ColumnMajor;
  } else if (a == LayoutEnum::ColumnMajor) {
    return LayoutEnum::RowMajor;
  }
  return LayoutEnum::Invalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Gemm
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm(Manifest &manifest, int64_t m, int64_t n, int64_t k,
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

  typename Operation::OperationTrait trait;
  if (layout_c == LayoutEnum::RowMajor) {
    trait = {OperationEnum::Matmul,
             EpilogueEnum::None,
             cutlass_type_to_dtype_v<ElementInputA>,
             layout_a,
             cutlass_type_to_dtype_v<ElementInputB>,
             layout_b,
             cutlass_type_to_dtype_v<ElementOutput>,
             LayoutEnum::RowMajor,
             cutlass_type_to_dtype_v<ElementAccumulator>};
  } else if (layout_c == LayoutEnum::ColumnMajor) {
    trait = {OperationEnum::Matmul,
             EpilogueEnum::None,
             cutlass_type_to_dtype_v<ElementInputB>,
             transpose_matrix(layout_b),
             cutlass_type_to_dtype_v<ElementInputA>,
             transpose_matrix(layout_a),
             cutlass_type_to_dtype_v<ElementOutput>,
             LayoutEnum::RowMajor,
             cutlass_type_to_dtype_v<ElementAccumulator>};
  }
  std::vector<Result> results;
  for (auto &kernel : manifest.kernels) {
    if (kernel->Trait() != trait) {
      continue;
    }
    if (layout_c == LayoutEnum::RowMajor) {
      kernel->SetArgument(m, n, k, (void *)a, (void *)b, nullptr, (void *)c, 1,
                          1.f, 0.f);
    } else {
      kernel->SetArgument(n, m, k, (void *)b, (void *)a, nullptr, (void *)c, 1,
                          1.f, 0.f);
    }
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream, nullptr);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(c, ref_c, m * n, 1e-3f, 1e-2f);
    float time = benchmark<Operation>(kernel, stream);
#if DEBUG
    std::cerr << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    std::cerr << ", " << time << ", " << cublas_time << "\n";
#endif
    results.push_back({kernel->Name(), passed, time, cublas_time});
  }
  std::cerr << "\n";

  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
  delete op;
  CUBLASCHECK(cublasDestroy(handle));

  if (results.size() == 0)
    return;
  std::sort(results.begin(), results.end());
  std::cout << "Gemm " << m << "x" << n << "x" << k << " "
            << layout_enum_to_str(layout_a) << "-"
            << layout_enum_to_str(layout_b) << "-"
            << layout_enum_to_str(layout_c) << ":\n";
  std::cout << "KernelName, Passed, CutlassTime, CublasTime\n";
  std::cout << results[0].kernel_name << ", "
            << (results[0].passed ? "Passed" : "Failed") << ", "
            << results[0].cutlass_time << ", " << results[0].library_time
            << "\n\n\n";
}

template void profile_gemm<__half, __half, float, float>(Manifest &, int64_t,
                                                         int64_t, int64_t,
                                                         LayoutEnum, LayoutEnum,
                                                         LayoutEnum);
template void profile_gemm<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void profile_gemm<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmBias
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm_bias(Manifest &manifest, int64_t m, int64_t n, int64_t k,
                       LayoutEnum layout_a, LayoutEnum layout_b,
                       LayoutEnum layout_c,
                       EpilogueEnum epilogue /* = EpilogueEnum::None */) {
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  TA *a = nullptr;
  TB *b = nullptr;
  TC *bias = nullptr;
  TC *d = nullptr;
  TC *ref_d = nullptr;
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(TA)));
  CUDACHECK(cudaMalloc(&b, n * k * sizeof(TB)));

  assert(layout_c == LayoutEnum::RowMajor);
  CUDACHECK(cudaMalloc(&bias, n * sizeof(TC))); // only RowMajor
  CUDACHECK(cudaMalloc(&d, m * n * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_d, m * n * sizeof(TC)));
  RandCUDABuffer(a, m * k, -1.f, 1.f);
  RandCUDABuffer(b, n * k, -1.f, 1.f);
  RandCUDABuffer(bias, n, -1.f, 1.f); // only RowMajor
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
  bias_add<TC><<<grid, block, 0, stream>>>(bias, ref_d, m, n);
  after_kernel_launch();
  if (epilogue == EpilogueEnum::Relu) {
    relu<TC><<<(m * n + 256 - 1) / 256, 256, 0, stream>>>(ref_d, m * n);
    after_kernel_launch();
  } else if (epilogue == EpilogueEnum::Sigmoid) {
    sigmoid<TC><<<(m * n + 256 - 1) / 256, 256, 0, stream>>>(ref_d, m * n);
    after_kernel_launch();
  }
  CUDACHECK(cudaDeviceSynchronize());

  typename Operation::OperationTrait trait{
      OperationEnum::MatmulBias,
      epilogue,
      cutlass_type_to_dtype_v<ElementInputA>,
      layout_a,
      cutlass_type_to_dtype_v<ElementInputB>,
      layout_b,
      cutlass_type_to_dtype_v<ElementOutput>,
      layout_c,
      cutlass_type_to_dtype_v<ElementAccumulator>};
  std::vector<Result> results;
  for (auto &kernel : manifest.kernels) {
    if (kernel->Trait() != trait) {
      continue;
    }
    kernel->SetArgument(m, n, k, (void *)a, (void *)b, (void *)bias, (void *)d,
                        1, 1.f, /*unused beta*/ 0.f);
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream, nullptr);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(d, ref_d, m * n, 1e-2f, 1e-2f);
    float time = benchmark<Operation>(kernel, stream);
#if DEBUG
    std::cerr << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    std::cerr << ", " << time << "\n";
#endif
    results.push_back({kernel->Name(), passed, time, 0.f});
  }
  std::cerr << "\n";

  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(bias));
  CUDACHECK(cudaFree(d));
  CUDACHECK(cudaFree(ref_d));
  delete op;
  CUBLASCHECK(cublasDestroy(handle));

  if (results.size() == 0)
    return;
  std::sort(results.begin(), results.end());
  std::cout << "GemmBias" << epilogue_enum_to_str(epilogue) << " " << m << "x"
            << n << "x" << k << " " << layout_enum_to_str(layout_a) << "-"
            << layout_enum_to_str(layout_b) << "-"
            << layout_enum_to_str(layout_c) << ":\n";
  std::cout << "KernelName, Passed, CutlassTime\n";
  std::cout << results[0].kernel_name << ", "
            << (results[0].passed ? "Passed" : "Failed") << ", "
            << results[0].cutlass_time << "\n\n\n";
}

template void
profile_gemm_bias<__half, __half, float, float>(Manifest &, int64_t, int64_t,
                                                int64_t, LayoutEnum, LayoutEnum,
                                                LayoutEnum, EpilogueEnum);
template void profile_gemm_bias<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum);
template void profile_gemm_bias<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum);

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmGemm
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm_gemm(Manifest &manifest, int64_t m, int64_t n, int64_t k,
                       LayoutEnum layout_a, LayoutEnum layout_b,
                       LayoutEnum layout_c) {
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  TA *a = nullptr;
  TB *b = nullptr;
  TC *b1 = nullptr;
  TC *d = nullptr;
  TC *ref_d = nullptr;
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(TA)));
  CUDACHECK(cudaMalloc(&b, n * k * sizeof(TB)));
  CUDACHECK(cudaMalloc(&b1, m * n * sizeof(TC)));
  CUDACHECK(cudaMalloc(&d, m * n * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_d, m * n * sizeof(TC)));
  RandCUDABuffer(a, m * k, -1.f, 1.f);
  RandCUDABuffer(b, n * k, -1.f, 1.f);
  RandCUDABuffer(b1, m * n, -1.f, 1.f);
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
  cutlass::device_memory::allocation<uint8_t> workspace(m * n * sizeof(TC));
  op->Run(a, b, (TC *)workspace.get());
  op->Run((TC *)workspace.get(), b1, ref_d);
  CUDACHECK(cudaDeviceSynchronize());

  typename Operation::OperationTrait trait;
  typename Operation::OperationTrait trait1;
  trait = {OperationEnum::Matmul,
           EpilogueEnum::None,
           cutlass_type_to_dtype_v<ElementInputA>,
           layout_a,
           cutlass_type_to_dtype_v<ElementInputB>,
           layout_b,
           cutlass_type_to_dtype_v<ElementOutput>,
           LayoutEnum::RowMajor,
           cutlass_type_to_dtype_v<ElementAccumulator>};
  trait1 = {OperationEnum::Matmul,
            EpilogueEnum::None,
            cutlass_type_to_dtype_v<ElementOutput>,
            LayoutEnum::RowMajor,
            cutlass_type_to_dtype_v<ElementInputB>,
            layout_b,
            cutlass_type_to_dtype_v<ElementOutput>,
            LayoutEnum::RowMajor,
            cutlass_type_to_dtype_v<ElementAccumulator>};
  for (auto &kernel : manifest.kernels) {
    if (kernel->Trait() != trait || kernel->Trait1() != trait1) {
      continue;
    }
    kernel->SetArgument(m, n, k, (void *)a, (void *)b, (void *)b1, (void *)d, 1,
                        1.f, 0.f);
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream, nullptr);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(d, ref_d, m * n, 1e-3f, 1e-2f);
#if DEBUG
    std::cerr << kernel->Name() << ", " << (passed ? "Passed" : "Failed")
              << "\n";
#endif
  }

  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(b1));
  CUDACHECK(cudaFree(d));
  CUDACHECK(cudaFree(ref_d));
  delete op;
  CUBLASCHECK(cublasDestroy(handle));
}

template void profile_gemm_gemm<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void profile_gemm_gemm<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2d
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d(Manifest &manifest, int64_t N, int64_t iH, int64_t iW,
                    int64_t iC, int64_t oH, int64_t oW, int64_t oC, int64_t kH,
                    int64_t kW, int64_t strideH, int64_t strideW,
                    int64_t paddingH, int64_t paddingW,
                    int64_t dilationH /* = 1*/, int64_t dilationW /* = 1*/) {
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
  RandCUDABuffer(input, N * iH * iW * iC);   //, -1.f, 1.f);
  RandCUDABuffer(filter, oC * kH * kW * iC); //, -1.f, 1.f);
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
      OperationEnum::Conv2d,
      EpilogueEnum::None,
      cutlass_type_to_dtype_v<ElementInputA>,
      LayoutEnum::NHWC,
      cutlass_type_to_dtype_v<ElementInputB>,
      LayoutEnum::NHWC,
      cutlass_type_to_dtype_v<ElementOutput>,
      LayoutEnum::NHWC,
      cutlass_type_to_dtype_v<ElementAccumulator>};
  std::vector<Result> results;
  for (auto kernel : manifest.kernels) {
    if (kernel->Trait() != trait) {
      continue;
    }
    kernel->SetArgument(N, iH, iW, iC, oH, oW, oC, kH, kW, strideH, strideW,
                        paddingH, paddingW, dilationH, dilationW, (void *)input,
                        (void *)filter, nullptr, (void *)output, 1, 1.f, 0.f);
    if (!kernel->Check()) {
      continue;
    }
    cutlass::device_memory::allocation<uint8_t> workspace(
        kernel->GetWorkspaceSize());
    kernel->Initialize(stream, workspace.get());
    kernel->Run();
    bool passed =
        CheckCUDABuffer<TC>(output, ref_output, N * oH * oW * oC, 1e-3f, 1e-2f);
    float time = benchmark<Operation>(kernel, stream);
#if DEBUG
    std::cerr << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    std::cerr << ", " << time << ", " << cudnn_time << "\n";
#endif
    results.push_back({kernel->Name(), passed, time, cudnn_time});
  }
  std::cerr << "\n";

  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(filter));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFree(ref_output));
  delete op;
  CUDNNCHECK(cudnnDestroy(handle));

  if (results.size() == 0)
    return;
  std::sort(results.begin(), results.end());
  std::cout << "Conv2dFrop " << N << "x" << iH << "x" << iW << "x" << iC << ", "
            << oC << "x" << kH << "x" << kW << "x" << iC << ", " << N << "x"
            << oH << "x" << oW << "x" << oC << ", "
            << "stride: " << strideH << "x" << strideW
            << ", padding: " << paddingH << "x" << paddingW << ":\n";
  std::cout << "KernelName, Passed, CutlassTime, CudnnTime\n";
  std::cout << results[0].kernel_name << ", "
            << (results[0].passed ? "Passed" : "Failed") << ", "
            << results[0].cutlass_time << ", " << results[0].library_time
            << "\n\n\n";
}

template void profile_conv2d<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template void profile_conv2d<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2dBias
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d_bias(Manifest &manifest, int64_t N, int64_t iH, int64_t iW,
                         int64_t iC, int64_t oH, int64_t oW, int64_t oC,
                         int64_t kH, int64_t kW, int64_t strideH,
                         int64_t strideW, int64_t paddingH, int64_t paddingW,
                         int64_t dilationH /* = 1*/, int64_t dilationW /* = 1*/,
                         EpilogueEnum epilogue /* = EpilogueEnum::None */) {
  // only profile NHWC layout
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  TA *input = nullptr;
  TB *filter = nullptr;
  TC *bias = nullptr;
  TC *output = nullptr;
  TC *ref_output = nullptr;
  CUDACHECK(cudaMalloc(&input, N * iH * iW * iC * sizeof(TA)));
  CUDACHECK(cudaMalloc(&filter, oC * kH * kW * iC * sizeof(TB)));
  CUDACHECK(cudaMalloc(&bias, oC * sizeof(TC)));
  CUDACHECK(cudaMalloc(&output, N * oH * oW * oC * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_output, N * oH * oW * oC * sizeof(TC)));
  RandCUDABuffer(input, N * iH * iW * iC, -1.f, 1.f);
  RandCUDABuffer(filter, oC * kH * kW * iC, -1.f, 1.f);
  RandCUDABuffer(bias, oC);
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
  dim3 block(16, 16);
  dim3 grid((oC + block.x - 1) / block.x,
            (N * oH * oW + block.y - 1) / block.y);
  bias_add<TC><<<grid, block, 0, stream>>>(bias, ref_output, N * oH * oW, oC);
  after_kernel_launch();
  if (epilogue == EpilogueEnum::Relu) {
    relu<TC><<<(N * oH * oW * oC + 256 - 1) / 256, 256, 0, stream>>>(
        ref_output, N * oH * oW * oC);
    after_kernel_launch();
  } else if (epilogue == EpilogueEnum::Sigmoid) {
    sigmoid<TC><<<(N * oH * oW * oC + 256 - 1) / 256, 256, 0, stream>>>(
        ref_output, N * oH * oW * oC);
    after_kernel_launch();
  }
  CUDACHECK(cudaDeviceSynchronize());

  typename Operation::OperationTrait trait{
      OperationEnum::Conv2dBias,
      epilogue,
      cutlass_type_to_dtype_v<ElementInputA>,
      LayoutEnum::NHWC,
      cutlass_type_to_dtype_v<ElementInputB>,
      LayoutEnum::NHWC,
      cutlass_type_to_dtype_v<ElementOutput>,
      LayoutEnum::NHWC,
      cutlass_type_to_dtype_v<ElementAccumulator>};
  std::vector<Result> results;
  for (auto kernel : manifest.kernels) {
    if (kernel->Trait() != trait) {
      continue;
    }
    kernel->SetArgument(N, iH, iW, iC, oH, oW, oC, kH, kW, strideH, strideW,
                        paddingH, paddingW, dilationH, dilationW, (void *)input,
                        (void *)filter, (void *)bias, (void *)output, 1, 1.f,
                        /*unused beta*/ 0.f);
    if (!kernel->Check()) {
      continue;
    }
    cutlass::device_memory::allocation<uint8_t> workspace(
        kernel->GetWorkspaceSize());
    kernel->Initialize(stream, workspace.get());
    kernel->Run();
    bool passed =
        CheckCUDABuffer<TC>(output, ref_output, N * oH * oW * oC, 1e-3f, 1e-2f);
    float time = benchmark<Operation>(kernel, stream);
#if DEBUG
    std::cerr << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    std::cerr << ", " << time << std::endl;
#endif
    results.push_back({kernel->Name(), passed, time, 0.f});
  }
  std::cerr << "\n";

  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(filter));
  CUDACHECK(cudaFree(bias));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFree(ref_output));
  delete op;
  CUDNNCHECK(cudnnDestroy(handle));

  if (results.size() == 0)
    return;
  std::sort(results.begin(), results.end());
  std::cout << "Conv2dFpropBias" << epilogue_enum_to_str(epilogue) << " " << N
            << "x" << iH << "x" << iW << "x" << iC << ", " << oC << "x" << kH
            << "x" << kW << "x" << iC << ", " << N << "x" << oH << "x" << oW
            << "x" << oC << ", "
            << "stride: " << strideH << "x" << strideW
            << ", padding: " << paddingH << "x" << paddingW << ":\n";
  std::cout << "KernelName, Passed, CutlassTime\n";
  std::cout << results[0].kernel_name << ", "
            << (results[0].passed ? "Passed" : "Failed") << ", "
            << results[0].cutlass_time << "\n\n\n";
}

template void profile_conv2d_bias<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    EpilogueEnum);
template void profile_conv2d_bias<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    EpilogueEnum);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2dConv2d
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d_conv2d(Manifest &manifest, int64_t N, int64_t iH,
                           int64_t iW, int64_t iC, int64_t oH, int64_t oW,
                           int64_t oC, int64_t kH, int64_t kW, int64_t strideH,
                           int64_t strideW, int64_t paddingH, int64_t paddingW,
                           int64_t dilationH /* = 1*/,
                           int64_t dilationW /* = 1*/) {}

template void profile_conv2d_conv2d<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template void profile_conv2d_conv2d<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
