#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "Operation.h"
#include "check.h"
#include "cutlass_dtype.h"
#include <iostream>

template <typename Gemm> class GemmOperation : public Operation {
public:
  using ElementA = typename Gemm::ElementA;
  using LayoutA = typename Gemm::LayoutA;
  using ElementB = typename Gemm::ElementB;
  using LayoutB = typename Gemm::LayoutB;
  using ElementC = typename Gemm::ElementC;
  using LayoutC = typename Gemm::LayoutC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  GemmOperation(const char *kernel_name,
                EpilogueEnum epilogue = EpilogueEnum::None)
      : Operation(kernel_name) {
    trait =
        Operation::OperationTrait{OperationEnum::Matmul,
                                  epilogue,
                                  cutlass_type_to_dtype_v<ElementA>,
                                  cutlass_layout_to_layout_v<LayoutA>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  GemmOperation(const char *kernel_name, OperationEnum op_enum,
                EpilogueEnum epilogue)
      : Operation(kernel_name) {
    trait =
        Operation::OperationTrait{op_enum,
                                  epilogue,
                                  cutlass_type_to_dtype_v<ElementA>,
                                  cutlass_layout_to_layout_v<LayoutA>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c, void *d, int64_t split_k_slices,
                           float alpha, float beta) override {
    typename cutlass::gemm::GemmCoord problem_size(m, n, k);
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
                 {(ElementC *)d, layoutC},
                 {(ElementAccumulator)alpha, (ElementAccumulator)beta},
                 split_k_slices};
  }

  virtual bool Check() override {
    return gemm.can_implement(arguments) == cutlass::Status::kSuccess;
  }

  virtual int64_t GetWorkspaceSize() override { return 0; }

  virtual void Initialize(cudaStream_t stream, void *workspace) override {
    CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  }

  virtual void Run() override { CUTLASS_CHECK(gemm()); }

  virtual const OperationTrait *Trait() override { return &trait; }

private:
  Gemm gemm;
  typename Operation::OperationTrait trait;

protected:
  typename Gemm::Arguments arguments;
};
