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

  virtual void SetArgument(int64_t m0, int64_t n0, int64_t k0, int64_t m1,
                           int64_t n1, int64_t k1, void *a0, void *b0, void *c0,
                           void *b1, void *c1, void *d1, int64_t split_k_slices,
                           float alpha0, float beta0, float alpha1,
                           float beta1) override {
    typename cutlass::gemm::GemmCoord problem_size_0(m0, n0, k1);
    typename cutlass::gemm::GemmCoord problem_size_1(m1, n1, k1);
    assert(cutlass_layout_to_layout_v<LayoutC> == LayoutEnum::RowMajor);
    assert(m0 == m1 && n0 == k1);
    LayoutA layoutA0(k0);
    LayoutB layoutB0(n0);
    LayoutC layoutC0(n0);
    LayoutB layoutB1(n1);
    LayoutC layoutC1(n1);
    LayoutC layoutD1(n1);
    if (cutlass_layout_to_layout_v<LayoutA> == LayoutEnum::ColumnMajor) {
      layoutA0 = LayoutA(m0);
    }
    if (cutlass_layout_to_layout_v<LayoutB> == LayoutEnum::ColumnMajor) {
      layoutB0 = LayoutB(k0);
      layoutB1 = LayoutB(k1);
    }
    // if (cutlass_layout_to_layout_v<LayoutC> == LayoutEnum::ColumnMajor) {
    //   layoutC = LayoutC(m0);
    // }
    arguments = {problem_size_0,
                 problem_size_1,
                 {(ElementA *)a0, layoutA0},
                 {(ElementB *)b0, layoutB0},
                 {(ElementC *)c0, layoutC0},
                 {(ElementB *)b1, layoutB1},
                 {(ElementC *)c1, layoutC1},
                 {(ElementC *)d1, layoutD1},
                 {(ElementAccumulator)alpha0, (ElementAccumulator)beta0},
                 {(ElementAccumulator)alpha1, (ElementAccumulator)beta1}};
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

  virtual const OperationTrait *Trait1() override { return &trait1; }

private:
  GemmGemm gemm;
  typename Operation::OperationTrait trait;
  typename Operation::OperationTrait trait1;
  typename GemmGemm::Arguments arguments;
};