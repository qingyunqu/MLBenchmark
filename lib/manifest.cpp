#include "manifest.h"
#include "Operation.h"
#include "benchmark.h"
#include "check.h"
#include "cutlass_dtype.h"
#include "matmul/CublasMatmul.h"
#include "matmul/util.h"

#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"

template <typename TA, typename TB, typename TC, typename CompOn>
void Manifest::profile(int64_t m, int64_t n, int64_t k, LayoutEnum layout_a,
                       LayoutEnum layout_b, LayoutEnum layout_c) {
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
  delete op;
  CUBLASCHECK(cublasDestroy(handle));

  typename Operation::OperationTrait trait{
      cutlass_type_to_dtype_v<ElementInputA>,     layout_a,
      cutlass_type_to_dtype_v<ElementInputB>,     layout_b,
      cutlass_type_to_dtype_v<ElementOutput>,     layout_c,
      cutlass_type_to_dtype_v<ElementAccumulator>};
  for (auto &kernel : kernels) {
    if (kernel->Trait() != trait) {
      continue;
    }
    kernel->SetArgument(m, n, k, (void *)a, (void *)b, (void *)c);
    if (!kernel->Check()) {
      continue;
    }
    kernel->Initialize(stream);
    kernel->Run();
    bool passed = CheckCUDABuffer<TC>(c, ref_c, m * n, 1e-2f);
    std::cout << kernel->Name() << ", " << (passed ? "Passed" : "Failed");
    float time = benchmark<Operation>(kernel, stream);
    std::cout << ", " << time << std::endl;
  }
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
}

template void Manifest::profile<__half, __half, float, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile<__half, __half, __half, float>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);
template void Manifest::profile<__half, __half, __half, __half>(
    int64_t, int64_t, int64_t, LayoutEnum, LayoutEnum, LayoutEnum);