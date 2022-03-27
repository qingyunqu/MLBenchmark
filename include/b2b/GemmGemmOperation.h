#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "Operation.h"
#include "check.h"
#include "cutlass_dtype.h"
#include <iostream>

template <typename GemmGemm> class GemmGemmOperation : public Operation {
public:
  using ElementA = typename GemmGemm::ElementA;
  using LayoutA = typename GemmGemm::LayoutA;
  using ElementB = typename GemmGemm::ElementB;
  using LayoutB = typename GemmGemm::LayoutB;
  using ElementC = typename GemmGemm::ElementC;
  using LayoutC = typename GemmGemm::LayoutC;
  using ElementAccumulator = typename GemmGemm::ElementAccumulator;

  GemmGemmOperation(const char *kernel_name) : Operation(kernel_name) {
    trait =
        Operation::OperationTrait{OperationEnum::Matmul,
                                  EpilogueEnum::None,
                                  cutlass_type_to_dtype_v<ElementA>,
                                  cutlass_layout_to_layout_v<LayoutA>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
    trait1 =
        Operation::OperationTrait{OperationEnum::Matmul,
                                  EpilogueEnum::None,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b0,
                           void *b1, void *d, int64_t /*split_k_slices*/,
                           float alpha, float beta) override {
    typename cutlass::gemm::GemmCoord problem_size_0(m, n, k);
    typename cutlass::gemm::GemmCoord problem_size_1(m, n, k);
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
    arguments = {problem_size_0,
                 problem_size_1,
                 {(ElementA *)a, layoutA},
                 {(ElementB *)b0, layoutB},
                 {nullptr, layoutC},
                 {(ElementB *)b1, layoutB},
                 {nullptr, layoutC},
                 {(ElementC *)d, layoutC},
                 {(ElementAccumulator)alpha, (ElementAccumulator)beta},
                 {(ElementAccumulator)alpha, (ElementAccumulator)beta}};
  }

  virtual bool Check() override {
    return gemm.can_implement(arguments) == cutlass::Status::kSuccess;
  }

  virtual int64_t GetWorkspaceSize() override { return 0; }

  virtual void Initialize(cudaStream_t stream, void *workspace) override {
    CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  }

  virtual void Run() override { CUTLASS_CHECK(gemm()); }

  virtual const OperationTrait &Trait() override { return trait; }

  virtual const OperationTrait &Trait1() override { return trait1; }

private:
  GemmGemm gemm;
  typename Operation::OperationTrait trait;
  typename Operation::OperationTrait trait1;
  typename GemmGemm::Arguments arguments;
};