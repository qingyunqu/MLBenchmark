#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "../check.h"
#include "../cutlass_dtype.h"
#include "../ops.h"
#include "../util.h"
#include "./cublas_matmul.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// Gemm operator cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8
using Operation_cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8 =
    cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2, 8, 8,
        false, cutlass::arch::OpMultiplyAdd

        >;

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

// Gemm operator cutlass_tensorop_s1688gemm_f16_64x128_32x2_nt_align8
using Operation_cutlass_tensorop_s1688gemm_f16_64x128_32x2_nt_align8 =
    cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2, 8, 8,
        false, cutlass::arch::OpMultiplyAdd

        >;

///////////////////////////////////////////////////////////////////////////////////////////////////

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
      return element_a == trait.element_a && element_b == trait.element_b &&
             element_c == trait.element_c && layout_a == trait.layout_a &&
             layout_b == trait.layout_b && layout_c == trait.layout_c &&
             accumulator == trait.accumulator;
    }
  };
  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c) = 0;
  virtual bool Check() = 0;
  virtual void Initialize(cudaStream_t) = 0;
  virtual void Run() = 0;
  virtual const char *Name() = 0;
  virtual const OperationTrait &Trait() = 0;
};

template <typename Gemm> class MatmulOperation : public Operation {
public:
  using ElementA = typename Gemm::ElementA;
  using LayoutA = typename Gemm::LayoutA;
  using ElementB = typename Gemm::ElementB;
  using LayoutB = typename Gemm::LayoutB;
  using ElementC = typename Gemm::ElementC;
  using LayoutC = typename Gemm::LayoutC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  MatmulOperation(const char *kernel_name) : kernel_name(kernel_name) {
    trait = {cutlass_type_to_dtype_v<ElementA>,
             cutlass_layout_to_layout_v<LayoutA>,
             cutlass_type_to_dtype_v<ElementB>,
             cutlass_layout_to_layout_v<LayoutB>,
             cutlass_type_to_dtype_v<ElementC>,
             cutlass_layout_to_layout_v<LayoutC>,
             cutlass_type_to_dtype_v<ElementAccumulator>};
  }
  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    LayoutA layoutA(k);
    LayoutB layoutB(n);
    LayoutC layoutC(n);
    if (cutlass_layout_to_layout_v<LayoutA> == LayoutEnum::ColumnMajor) {
      layoutA = LayoutA(m);
    }
    if (cutlass_layout_to_layout_v<LayoutB> == LayoutEnum::ColumnMajor) {
      layoutB = LayoutB(k);
    }
    if (cutlass_layout_to_layout_v<LayoutC> == LayoutEnum::ColumnMajor) {
      layoutC = LayoutC(m);
    }
    arguments = {problem_size,
                 {(ElementA *)a, layoutA},
                 {(ElementB *)b, layoutB},
                 {(ElementC *)c, layoutC},
                 {(ElementC *)c, layoutC},
                 {(ElementAccumulator)1, (ElementAccumulator)0},
                 /*split_k_slices=*/1};
  }
  virtual bool Check() {
    return gemm.can_implement(arguments) == cutlass::Status::kSuccess;
  }
  virtual void Initialize(cudaStream_t stream) {
    CUTLASS_CHECK(gemm.initialize(arguments, nullptr, stream));
  }
  virtual void Run() { CUTLASS_CHECK(gemm()); }
  virtual const char *Name() { return kernel_name; }
  virtual const OperationTrait &Trait() { return trait; }

private:
  const char *kernel_name;
  Gemm gemm;
  typename Gemm::Arguments arguments;
  typename Operation::OperationTrait trait;
};

namespace matmul {

class Manifest {
private:
  std::vector<Operation *> kernels;

public:
  void append(Operation *op) { kernels.push_back(op); }

  template <typename ElementInputA, typename LayoutInputA,
            typename ElementInputB, typename LayoutInputB,
            typename ElementOutput, typename LayoutOutput,
            typename ElementAccumulator>
  void append(Operation *op);

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

} // namesapce matmul