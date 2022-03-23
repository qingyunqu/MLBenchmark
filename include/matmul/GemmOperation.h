#pragma once

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "../check.h"
#include "../cutlass_dtype.h"
#include "../manifest.h"
#include "../ops.h"
#include "../util.h"

template <typename Gemm> class GemmOperation : public Operation {
public:
  using ElementA = typename Gemm::ElementA;
  using LayoutA = typename Gemm::LayoutA;
  using ElementB = typename Gemm::ElementB;
  using LayoutB = typename Gemm::LayoutB;
  using ElementC = typename Gemm::ElementC;
  using LayoutC = typename Gemm::LayoutC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  GemmOperation(const char *kernel_name) : Operation(kernel_name) {
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
  virtual const OperationTrait &Trait() { return trait; }

private:
  Gemm gemm;
  typename Gemm::Arguments arguments;
  typename Operation::OperationTrait trait;
};
