#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"

#include "check.h"
#include "cutlass_dtype.h"
#include "dtype.h"
#include "matmul/cublas_matmul.h"
#include "ops.h"
#include "util.h"

class Operation {
public:
  struct OperationTrait {
    DTypeEnum element_a;
    LayoutEnum layout_a;
    DTypeEnum element_b;
    LayoutEnum layout_b;
    DTypeEnum element_c;
    LayoutEnum layout_c;
    DTypeEnum accumulator;
    bool operator!=(const OperationTrait &trait) const {
      return !(element_a == trait.element_a && element_b == trait.element_b &&
               element_c == trait.element_c && layout_a == trait.layout_a &&
               layout_b == trait.layout_b && layout_c == trait.layout_c &&
               accumulator == trait.accumulator);
    }
  };

  Operation(const char *kernel_name) : kernel_name(kernel_name) {}
  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c) = 0;
  virtual bool Check() = 0;
  virtual void Initialize(cudaStream_t) = 0;
  virtual void Run() = 0;
  const char *Name() { return kernel_name; }
  virtual const OperationTrait &Trait() = 0;

private:
  const char *kernel_name;
};

class Manifest {
private:
  std::vector<Operation *> kernels;

public:
  void reserve(size_t count) { kernels.reserve(count); }

  void append(Operation *op) { kernels.push_back(op); }

  template <typename TA, typename TB, typename TC>
  void init_tensor(int64_t m, int64_t n, int64_t k, TA *&a, TB *&b, TC *&c,
                   TC *&ref_c) {
    CUDACHECK(cudaMalloc(&a, m * k * sizeof(TA)));
    CUDACHECK(cudaMalloc(&b, n * k * sizeof(TB)));
    CUDACHECK(cudaMalloc(&c, m * n * sizeof(TC)));
    CUDACHECK(cudaMalloc(&ref_c, m * n * sizeof(TC)));
    RandCUDABuffer(a, m * k);
    RandCUDABuffer(b, n * k);
    RandCUDABuffer(c, m * n);
    FillCUDABuffer(ref_c, m * n);
  }

  template <typename ElementInputA, typename ElementInputB,
            typename ElementOutput, typename ElementAccumulator>
  void profile(int64_t m, int64_t n, int64_t k, LayoutEnum layout_a,
               LayoutEnum layout_b, LayoutEnum layout_c) {
    using TA = typename cutlass_type_to_ctype<ElementInputA>::type;
    using TB = typename cutlass_type_to_ctype<ElementInputB>::type;
    using TC = typename cutlass_type_to_ctype<ElementOutput>::type;
    TA *a = nullptr;
    TB *b = nullptr;
    TC *c = nullptr;
    TC *ref_c = nullptr;
    init_tensor(m, n, k, a, b, c, ref_c);

    cudaStream_t stream = nullptr;
    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    CUBLASCHECK(cublasSetStream(handle, stream));
    Matmul<TA, TC> *op = new CublasMatmul<TA, TC, ElementAccumulator>(
        m, n, k, layout_a == LayoutEnum::ColumnMajor,
        layout_b == LayoutEnum::ColumnMajor,
        layout_c == LayoutEnum::ColumnMajor, handle);
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
      bool passed = CheckCUDABuffer<TC>(c, ref_c, m * n, 1e-5f);
      std::cout << kernel->Name() << " : " << (passed ? "Passed" : "Failed")
                << std::endl;
    }
    CUDACHECK(cudaFree(a));
    CUDACHECK(cudaFree(b));
    CUDACHECK(cudaFree(c));
    CUDACHECK(cudaFree(ref_c));
  }

  ~Manifest() {
    for (auto &kernel : kernels) {
      delete kernel;
    }
  }
};
