#include "Manifest.h"
#include "Operation.h"
#include "benchmark.h"
#include "check.h"
#include "convolution/CudnnConv.h"
#include "cutlass_dtype.h"
#include "matmul/CublasMatmul.h"
#include "profile.h"
#include "util/kernel.h"
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

void print_operation_trait(std::ostream &out, Operation::OperationTrait trait) {
  out << operation_enum_to_str(trait.operation)
      << epilogue_enum_to_str(trait.epilogue) << "    ";
  out << dtype_enum_to_str(trait.element_a) << "-"
      << dtype_enum_to_str(trait.element_b) << "-"
      << dtype_enum_to_str(trait.element_c) << "-"
      << dtype_enum_to_str(trait.accumulator) << "    ";
  out << layout_enum_to_str(trait.layout_a) << "-"
      << layout_enum_to_str(trait.layout_b) << "-"
      << layout_enum_to_str(trait.layout_c) << "\n";
}

Operation *get_kernel_by_name(Manifest &manifest, const std::string &name) {
  for (auto kernel : manifest.kernels) {
    if (kernel->Name() == name) {
      return kernel;
    }
  }
  return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Gemm
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm(Manifest &manifest, int64_t m, int64_t n, int64_t k,
                  LayoutEnum layout_a, LayoutEnum layout_b, LayoutEnum layout_c,
                  EpilogueEnum epilogue,
                  const std::unordered_set<std::string> &run_kernels) {
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
  auto *op = new CublasMatmul<TA, TC, CompOn>(
      m, n, k, layout_a == LayoutEnum::ColumnMajor,
      layout_b == LayoutEnum::ColumnMajor, layout_c == LayoutEnum::ColumnMajor,
      handle);
  op->SetArgument(a, b, ref_c);
  op->Run();
  auto *act = new MyActivation<TC>(m * n, epilogue, stream);
  act->SetArgument(ref_c);
  act->Run();
  CUDACHECK(cudaDeviceSynchronize());
  float cublas_time = benchmark2(op, act, stream);
#if DEBUG
  std::cerr << "Cublas"
            << " + " << epilogue_enum_to_str(epilogue)
            << " Time: " << cublas_time << "\n";
#endif

  typename Operation::OperationTrait trait;
  if (layout_c == LayoutEnum::RowMajor) {
    trait = {OperationEnum::Matmul,
             epilogue,
             cutlass_type_to_dtype_v<ElementInputA>,
             layout_a,
             cutlass_type_to_dtype_v<ElementInputB>,
             layout_b,
             cutlass_type_to_dtype_v<ElementOutput>,
             LayoutEnum::RowMajor,
             cutlass_type_to_dtype_v<ElementAccumulator>};
  } else if (layout_c == LayoutEnum::ColumnMajor) {
    trait = {OperationEnum::Matmul,
             epilogue,
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
    if (run_kernels.size() != 0 &&
        run_kernels.find(kernel->Name()) == run_kernels.end()) {
      continue;
    }
    if (*(kernel->Trait()) != trait) {
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
  delete act;
  CUBLASCHECK(cublasDestroy(handle));

  if (results.size() == 0)
    return;
  std::sort(results.begin(), results.end());
  std::cout << "Gemm" << epilogue_enum_to_str(epilogue) << " " << m << "x" << n
            << "x" << k << " " << layout_enum_to_str(layout_a) << "-"
            << layout_enum_to_str(layout_b) << "-"
            << layout_enum_to_str(layout_c) << ":\n";
  std::cout << "KernelName, Passed, CutlassTime, CublasTime\n";
  std::cout << results[0].kernel_name << ", "
            << (results[0].passed ? "Passed" : "Failed") << ", "
            << results[0].cutlass_time << ", " << cublas_time << "\n\n\n";
}

template void profile_gemm<__half, __half, float, float>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum, const std::unordered_set<std::string> &);
template void profile_gemm<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum, const std::unordered_set<std::string> &);
template void profile_gemm<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum, const std::unordered_set<std::string> &);

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmBias
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm_bias(Manifest &manifest, int64_t m, int64_t n, int64_t k,
                       LayoutEnum layout_a, LayoutEnum layout_b,
                       LayoutEnum layout_c, EpilogueEnum epilogue,
                       const std::unordered_set<std::string> &run_kernels) {
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
  auto *op = new CublasMatmul<TA, TC, CompOn>(
      m, n, k, layout_a == LayoutEnum::ColumnMajor,
      layout_b == LayoutEnum::ColumnMajor, layout_c == LayoutEnum::ColumnMajor,
      handle);
  op->SetArgument(a, b, ref_d);
  op->Run();
  BiasAdd(bias, ref_d, m, n, stream);
  auto *act = new MyActivation<TC>(m * n, epilogue, stream);
  act->SetArgument(ref_d);
  act->Run();
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
    if (run_kernels.size() != 0 &&
        run_kernels.find(kernel->Name()) == run_kernels.end()) {
      continue;
    }
    if (*(kernel->Trait()) != trait) {
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
  delete act;
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

template void profile_gemm_bias<__half, __half, float, float>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum, const std::unordered_set<std::string> &);
template void profile_gemm_bias<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum, const std::unordered_set<std::string> &);
template void profile_gemm_bias<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum,
    EpilogueEnum, const std::unordered_set<std::string> &);

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmGemm
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm_gemm(Manifest &manifest, int64_t m0, int64_t n0, int64_t k0,
                       int64_t n1, LayoutEnum layout_a, LayoutEnum layout_b,
                       LayoutEnum layout_c) {
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  assert(layout_c == LayoutEnum::RowMajor);
  int64_t m1 = m0, k1 = n0;
  TA *a0 = nullptr;
  TB *b0 = nullptr;
  TB *b1 = nullptr;
  TC *d1 = nullptr;
  TC *ref_d0 = nullptr;
  TC *ref_d1 = nullptr;
  CUDACHECK(cudaMalloc(&a0, m0 * k0 * sizeof(TA)));
  CUDACHECK(cudaMalloc(&b0, n0 * k0 * sizeof(TB)));
  CUDACHECK(cudaMalloc(&b1, n1 * k1 * sizeof(TB)));
  CUDACHECK(cudaMalloc(&d1, m1 * n1 * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_d0, m0 * n0 * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_d1, m1 * n1 * sizeof(TC)));
  RandCUDABuffer(a0, m0 * k0, -1.f, 1.f);
  RandCUDABuffer(b0, n0 * k0, -1.f, 1.f);
  RandCUDABuffer(b1, n1 * k1, -1.f, 1.f);
  RandCUDABuffer(d1, m1 * n1);
  FillCUDABuffer(ref_d1, m1 * n1);

  cudaStream_t stream = nullptr;
  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  {
    auto *op0 = new CublasMatmul<TA, TC, CompOn>(
        m0, n0, k0, layout_a == LayoutEnum::ColumnMajor,
        layout_b == LayoutEnum::ColumnMajor,
        layout_c == LayoutEnum::ColumnMajor, handle);
    op0->SetArgument(a0, b0, ref_d0);
    op0->Run();
    float time0 = benchmark(op0, stream);
    delete op0;

    auto *op1 = new CublasMatmul<TC, TC, CompOn>(
        m1, n1, k1, layout_c == LayoutEnum::ColumnMajor,
        layout_b == LayoutEnum::ColumnMajor,
        layout_c == LayoutEnum::ColumnMajor, handle);
    op1->SetArgument(ref_d0, b1, ref_d1);
    op1->Run();
    float time1 = benchmark(op1, stream);
    delete op1;

    std::cout << "cublas time0: " << time0 << "\n";
    std::cout << "cublas time1: " << time1 << "\n";
    std::cout << "cublas total time: " << time0 + time1 << "\n\n";
  }
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
  {
    auto kernel0 = get_kernel_by_name(manifest, "Gemm0");
    assert(*kernel0->Trait() == trait);
    cutlass::device_memory::allocation<uint8_t> temp_d0(m0 * n0 * sizeof(TC));
    kernel0->SetArgument(m0, n0, k0, (void *)a0, (void *)b0, nullptr,
                         temp_d0.get(), 1, 1.f, 0.f);
    assert(kernel0->Check());
    kernel0->Initialize(stream, nullptr);
    kernel0->Run();
    bool passed =
        CheckCUDABuffer<TC>((TC *)temp_d0.get(), ref_d0, m0 * n0, 1e-3f, 1e-2f);
    std::cout << kernel0->Name() << ", " << (passed ? "Passed" : "Failed")
              << ", ";
    float time0 = benchmark(kernel0, stream);
    std::cout << time0 << "\n";

    auto kernel1 = get_kernel_by_name(manifest, "Gemm1");
    assert(*kernel1->Trait() == trait);
    cutlass::device_memory::allocation<uint8_t> temp_d1(m1 * n1 * sizeof(TC));
    kernel1->SetArgument(m1, n1, k1, temp_d0.get(), (void *)b1, nullptr,
                         temp_d1.get(), 1, 1.f, 0.f);
    assert(kernel1->Check());
    kernel1->Initialize(stream, nullptr);
    kernel1->Run();
    passed =
        CheckCUDABuffer<TC>((TC *)temp_d1.get(), ref_d1, m1 * n1, 1e-3f, 1e-2f);
    std::cout << kernel1->Name() << ", " << (passed ? "Passed" : "Failed")
              << ", ";
    float time1 = benchmark(kernel1, stream);
    std::cout << time1 << "\n";

    std::cout << "Total Time: " << time0 + time1 << "\n\n";
  }
  for (auto &kernel : manifest.kernels) {
    if (kernel->Trait1() == nullptr) {
      continue;
    }
    if (*(kernel->Trait()) != trait || *(kernel->Trait1()) != trait1) {
      continue;
    }
    kernel->SetArgument(m0, n0, k0, m1, n1, k1, (void *)a0, (void *)b0, nullptr,
                        (void *)b1, nullptr, (void *)d1, 1, 1.f, 0.f, 1.f, 0.f);
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream, nullptr);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(d1, ref_d1, m1 * n1, 1e-3f, 1e-2f);
#if DEBUG
    std::cerr << kernel->Name() << ", " << (passed ? "Passed" : "Failed")
              << ", ";
#endif
    float time = benchmark(kernel, stream);
    std::cout << time << "\n\n";
  }

  CUDACHECK(cudaFree(a0));
  CUDACHECK(cudaFree(b0));
  CUDACHECK(cudaFree(b1));
  CUDACHECK(cudaFree(d1));
  CUDACHECK(cudaFree(ref_d0));
  CUDACHECK(cudaFree(ref_d1));
  CUBLASCHECK(cublasDestroy(handle));
}

template void
profile_gemm_gemm<__half, __half, __half, __half>(Manifest &, int64_t, int64_t,
                                                  int64_t, int64_t, LayoutEnum,
                                                  LayoutEnum, LayoutEnum);
template void
profile_gemm_gemm<__half, __half, __half, float>(Manifest &, int64_t, int64_t,
                                                 int64_t, int64_t, LayoutEnum,
                                                 LayoutEnum, LayoutEnum);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2d
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d(Manifest &manifest, int64_t N, int64_t iH, int64_t iW,
                    int64_t iC, int64_t oH, int64_t oW, int64_t oC, int64_t kH,
                    int64_t kW, int64_t strideH, int64_t strideW,
                    int64_t paddingH, int64_t paddingW, int64_t dilationH,
                    int64_t dilationW, EpilogueEnum epilogue,
                    const std::unordered_set<std::string> &run_kernels) {
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
  auto *op = new CudnnConv<TA, TC, CompOn>(
      "NHWC", N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH,
      paddingW, dilationH, dilationW, handle);
  op->AllocWorkspace();
  op->SetArgument(input, filter, ref_output);
  op->Run();
  auto *act = new MyActivation<TC>(N * oH * oW * oC, epilogue, stream);
  act->SetArgument(ref_output);
  act->Run();
  CUDACHECK(cudaDeviceSynchronize());
  float cudnn_time = benchmark2(op, act, stream);
#if DEBUG
  std::cerr << "Cudnn"
            << " + " << epilogue_enum_to_str(epilogue)
            << " Time: " << cudnn_time << "\n";
#endif

  typename Operation::OperationTrait trait{
      OperationEnum::Conv2d,
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
    if (run_kernels.size() != 0 &&
        run_kernels.find(kernel->Name()) == run_kernels.end()) {
      continue;
    }
    if (*(kernel->Trait()) != trait) {
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
    bool passed = CheckCUDABuffer<TC>(output, ref_output, N * oH * oW * oC,
                                      1e-3f, 1e-2f, 20);
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
  delete act;
  CUDNNCHECK(cudnnDestroy(handle));

  if (results.size() == 0)
    return;
  std::sort(results.begin(), results.end());
  std::cout << "Conv2dFrop" << epilogue_enum_to_str(epilogue) << " " << N << "x"
            << iH << "x" << iW << "x" << iC << ", " << oC << "x" << kH << "x"
            << kW << "x" << iC << ", " << N << "x" << oH << "x" << oW << "x"
            << oC << ", "
            << "stride: " << strideH << "x" << strideW
            << ", padding: " << paddingH << "x" << paddingW << ":\n";
  std::cout << "KernelName, Passed, CutlassTime, CudnnTime\n";
  std::cout << results[0].kernel_name << ", "
            << (results[0].passed ? "Passed" : "Failed") << ", "
            << results[0].cutlass_time << ", " << cudnn_time << "\n\n\n";
}

template void profile_conv2d<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    EpilogueEnum, const std::unordered_set<std::string> &);
template void profile_conv2d<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    EpilogueEnum, const std::unordered_set<std::string> &);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2dBias
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d_bias(Manifest &manifest, int64_t N, int64_t iH, int64_t iW,
                         int64_t iC, int64_t oH, int64_t oW, int64_t oC,
                         int64_t kH, int64_t kW, int64_t strideH,
                         int64_t strideW, int64_t paddingH, int64_t paddingW,
                         int64_t dilationH, int64_t dilationW,
                         EpilogueEnum epilogue,
                         const std::unordered_set<std::string> &run_kernels) {
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
  auto *op = new CudnnConvBias<TA, TC, CompOn>(
      "NHWC", N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH,
      paddingW, dilationH, dilationW, handle, EpilogueEnum::None);
  op->AllocWorkspace();
  op->SetArgument(input, filter, bias, ref_output);
  op->Run();
  // auto *act = new CudnnActivation<TC>({N, oH, oW, oC}, epilogue, handle);
  auto *act = new MyActivation<TC>(N * oH * oW * oC, epilogue, stream);
  act->SetArgument(ref_output);
  act->Run();
  CUDACHECK(cudaDeviceSynchronize());
  float cudnn_time = benchmark2(op, act, stream);
#if DEBUG
  std::cerr << "CudnnConvBias"
            << " + " << epilogue_enum_to_str(epilogue)
            << " Time: " << cudnn_time << "\n";
#endif

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
    if (run_kernels.size() != 0 &&
        run_kernels.find(kernel->Name()) == run_kernels.end()) {
      continue;
    }
    if (*(kernel->Trait()) != trait) {
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
  delete act;
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
  std::cout << "KernelName, Passed, CutlassTime, CudnnTime\n";
  std::cout << results[0].kernel_name << ", "
            << (results[0].passed ? "Passed" : "Failed") << ", "
            << results[0].cutlass_time << ", " << cudnn_time << "\n\n\n";
}

template void profile_conv2d_bias<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    EpilogueEnum, const std::unordered_set<std::string> &);
template void profile_conv2d_bias<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    EpilogueEnum, const std::unordered_set<std::string> &);

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2dConv2d
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d_conv2d(Manifest &manifest, int64_t N0, int64_t iH0,
                           int64_t iW0, int64_t iC0, int64_t oH0, int64_t oW0,
                           int64_t oC0, int64_t kH0, int64_t kW0,
                           int64_t strideH0, int64_t strideW0,
                           int64_t paddingH0, int64_t paddingW0,
                           int64_t dilationH0, int64_t dilationW0, int64_t N1,
                           int64_t iH1, int64_t iW1, int64_t iC1, int64_t oH1,
                           int64_t oW1, int64_t oC1, int64_t kH1, int64_t kW1,
                           int64_t strideH1, int64_t strideW1,
                           int64_t paddingH1, int64_t paddingW1,
                           int64_t dilationH1, int64_t dilationW1) {
  // only profile NHWC layout
  using ElementInputA = typename ctype_to_cutlass_type<TA>::type;
  using ElementInputB = typename ctype_to_cutlass_type<TB>::type;
  using ElementOutput = typename ctype_to_cutlass_type<TC>::type;
  using ElementAccumulator = typename ctype_to_cutlass_type<CompOn>::type;
  TA *input0 = nullptr;
  TB *filter0 = nullptr;
  TC *ref_output0 = nullptr;
  TB *filter1 = nullptr;
  TC *output1 = nullptr;
  TC *ref_output1 = nullptr;
  CUDACHECK(cudaMalloc(&input0, N0 * iH0 * iW0 * iC0 * sizeof(TA)));
  CUDACHECK(cudaMalloc(&filter0, oC0 * kH0 * kW0 * iC0 * sizeof(TB)));
  CUDACHECK(cudaMalloc(&ref_output0, N0 * oH0 * oW0 * oC0 * sizeof(TC)));
  CUDACHECK(cudaMalloc(&filter1, oC1 * kH1 * kW1 * iC1 * sizeof(TB)));
  CUDACHECK(cudaMalloc(&output1, N1 * oH1 * oW1 * oC1 * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_output1, N1 * oH1 * oW1 * oC1 * sizeof(TC)));
  RandCUDABuffer(input0, N0 * iH0 * iW0 * iC0, -1.f, 1.f);
  RandCUDABuffer(filter0, oC0 * kH0 * kW0 * iC0, -1.f, 1.f);
  RandCUDABuffer(ref_output0, N0 * oH0 * oW0 * oC0);
  RandCUDABuffer(filter1, oC1 * kH1 * kW1 * iC1, -1.f, 1.f);
  RandCUDABuffer(output1, N1 * oH1 * oW1 * oC1);
  RandCUDABuffer(ref_output1, N1 * oH1 * oW1 * oC1);

  cudaStream_t stream = nullptr;
  cudnnHandle_t handle;
  CUDNNCHECK(cudnnCreate(&handle));
  CUDNNCHECK(cudnnSetStream(handle, stream));
  {
    auto *op0 = new CudnnConv<TA, TC, CompOn>(
        "NHWC", N0, iC0, iH0, iW0, oC0, kH0, kW0, oH0, oW0, strideH0, strideW0,
        paddingH0, paddingW0, dilationH0, dilationW0, handle);
    op0->AllocWorkspace();
    op0->SetArgument(input0, filter0, ref_output0);
    op0->Run();
    float time0 = benchmark(op0, stream);
    delete op0;
    Relu(ref_output0, N0 * oH0 * oW0 * oC0, stream);
    CUDACHECK(cudaDeviceSynchronize());

    auto *op1 = new CudnnConv<TA, TC, CompOn>(
        "NHWC", N1, iC1, iH1, iW1, oC1, kH1, kW1, oH1, oW1, strideH1, strideW1,
        paddingH1, paddingW1, dilationH1, dilationW1, handle);
    op1->AllocWorkspace();
    op1->SetArgument(ref_output0, filter1, ref_output1);
    op1->Run();
    float time1 = benchmark(op1, stream);
    delete op1;
    Relu(ref_output1, N1 * oH1 * oW1 * oC1, stream);
    CUDACHECK(cudaDeviceSynchronize());

    std::cout << "cudnn time0: " << time0 << "\n";
    std::cout << "cudnn time1: " << time1 << "\n";
    std::cout << "cudnn total time: " << time0 + time1 << "\n\n";
  }
  CUDACHECK(cudaDeviceSynchronize());

  typename Operation::OperationTrait trait;
  typename Operation::OperationTrait trait1;
  trait = {OperationEnum::Conv2d,
           EpilogueEnum::None,
           cutlass_type_to_dtype_v<ElementInputA>,
           LayoutEnum::NHWC,
           cutlass_type_to_dtype_v<ElementInputB>,
           LayoutEnum::NHWC,
           cutlass_type_to_dtype_v<ElementOutput>,
           LayoutEnum::NHWC,
           cutlass_type_to_dtype_v<ElementAccumulator>};
  trait1 = {OperationEnum::Conv2d,
            EpilogueEnum::None,
            cutlass_type_to_dtype_v<ElementOutput>,
            LayoutEnum::NHWC,
            cutlass_type_to_dtype_v<ElementInputB>,
            LayoutEnum::NHWC,
            cutlass_type_to_dtype_v<ElementOutput>,
            LayoutEnum::NHWC,
            cutlass_type_to_dtype_v<ElementAccumulator>};
  {
    auto kernel0 = get_kernel_by_name(manifest, "Conv2d0");
    assert(kernel0 != nullptr);
    assert(*kernel0->Trait() == trait);
    cutlass::device_memory::allocation<uint8_t> temp_output0(N0 * oH0 * oW0 *
                                                             oC0 * sizeof(TC));
    kernel0->SetArgument(N0, iH0, iW0, iC0, oH0, oW0, oC0, kH0, kW0, strideH0,
                         strideW0, paddingH0, paddingW0, dilationH0, dilationW0,
                         (void *)input0, (void *)filter0, nullptr,
                         (void *)temp_output0.get(), 1, 1.f, 0.f);
    assert(kernel0->Check());
    cutlass::device_memory::allocation<uint8_t> workspace0(
        kernel0->GetWorkspaceSize());
    kernel0->Initialize(stream, workspace0.get());
    kernel0->Run();
    bool passed = CheckCUDABuffer<TC>((TC *)temp_output0.get(), ref_output0,
                                      N0 * oH0 * oW0 * oC0, 1e-2f, 1e-1f);
    std::cout << kernel0->Name() << ", " << (passed ? "Passed" : "Failed")
              << ", ";
    float time0 = benchmark(kernel0, stream);
    std::cout << time0 << "\n";

    auto kernel1 = get_kernel_by_name(manifest, "Conv2d1");
    assert(kernel1 != nullptr);
    assert(*kernel1->Trait() == trait1);
    cutlass::device_memory::allocation<uint8_t> temp_output1(N1 * oH1 * oW1 *
                                                             oC1 * sizeof(TC));
    kernel1->SetArgument(N1, iH1, iW1, iC1, oH1, oW1, oC1, kH1, kW1, strideH1,
                         strideW1, paddingH1, paddingW1, dilationH1, dilationW1,
                         (void *)temp_output0.get(), (void *)filter1, nullptr,
                         (void *)temp_output1.get(), 1, 1.f, 0.f);
    assert(kernel1->Check());
    cutlass::device_memory::allocation<uint8_t> workspace1(
        kernel1->GetWorkspaceSize());
    kernel1->Initialize(stream, workspace1.get());
    kernel1->Run();
    passed = CheckCUDABuffer<TC>((TC *)temp_output1.get(), ref_output1,
                                 N1 * oH1 * oW1 * oC1, 1e-2f, 1e-1f);
    std::cout << kernel1->Name() << ", " << (passed ? "Passed" : "Failed")
              << ", ";
    float time1 = benchmark(kernel1, stream);
    std::cout << time1 << "\n";

    std::cout << "Total Time: " << time0 + time1 << "\n\n";
  }

  {
    auto kernel = get_kernel_by_name(manifest, "b2b_conv2d");
    assert(kernel != nullptr);
    assert(*kernel->Trait() == trait && *kernel->Trait1() == trait1);
    cutlass::device_memory::allocation<uint8_t> scale0(oC0 * sizeof(CompOn));
    cutlass::device_memory::allocation<uint8_t> bias0(oC0 * sizeof(CompOn));
    FillCUDABuffer((CompOn *)scale0.get(), oC0, 1.f);
    FillCUDABuffer((CompOn *)bias0.get(), oC0, 0.f);
    kernel->SetArgument(
        N0, iH0, iW0, iC0, oH0, oW0, oC0, kH0, kW0, strideH0, strideW0,
        paddingH0, paddingW0, dilationH0, dilationW0, N1, iH1, iW1, iC1, oH1,
        oW1, oC1, kH1, kW1, strideH1, strideW1, paddingH1, paddingW1,
        dilationH1, dilationW1, (void *)input0, (void *)filter0,
        /*bias0*/ (void *)scale0.get(), (void *)filter1,
        /*bias1*/ (void *)bias0.get(), (void *)output1, 1, 1.f, 0.f, 1.f, 0.f);
    assert(kernel->Check());
    cutlass::device_memory::allocation<uint8_t> workspace(
        kernel->GetWorkspaceSize());
    kernel->Initialize(stream, workspace.get());
    kernel->Run();
    CUDACHECK(cudaDeviceSynchronize());
    bool passed = CheckCUDABuffer<TC>(output1, ref_output1,
                                      N1 * oH1 * oW1 * oC1, 1e-2f, 1e-1f, 10);
    std::cout << kernel->Name() << ", " << (passed ? "Passed" : "Failed")
              << ", ";
    float time = benchmark(kernel, stream);
    std::cout << time << "\n";
  }

  CUDACHECK(cudaFree(input0));
  CUDACHECK(cudaFree(filter0));
  CUDACHECK(cudaFree(ref_output0));
  CUDACHECK(cudaFree(filter1));
  CUDACHECK(cudaFree(output1));
  CUDACHECK(cudaFree(ref_output1));
}

template void profile_conv2d_conv2d<__half, __half, __half, float>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template void profile_conv2d_conv2d<__half, __half, __half, __half>(
    Manifest &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
