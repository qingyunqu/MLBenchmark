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

  GemmOperation(const char *kernel_name) : Operation(kernel_name) {
    trait = {OperationEnum::Matmul,
             cutlass_type_to_dtype_v<ElementA>,
             cutlass_layout_to_layout_v<LayoutA>,
             cutlass_type_to_dtype_v<ElementB>,
             cutlass_layout_to_layout_v<LayoutB>,
             cutlass_type_to_dtype_v<ElementC>,
             cutlass_layout_to_layout_v<LayoutC>,
             cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  GemmOperation(const char *kernel_name, OperationEnum op_enum)
      : Operation(kernel_name) {
    trait = {op_enum,
             cutlass_type_to_dtype_v<ElementA>,
             cutlass_layout_to_layout_v<LayoutA>,
             cutlass_type_to_dtype_v<ElementB>,
             cutlass_layout_to_layout_v<LayoutB>,
             cutlass_type_to_dtype_v<ElementC>,
             cutlass_layout_to_layout_v<LayoutC>,
             cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c, void *d) override {
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
                 {(ElementAccumulator)1, (ElementAccumulator)0},
                 /*split_k_slices=*/1};
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

private:
  Gemm gemm;
  typename Operation::OperationTrait trait;

protected:
  typename Gemm::Arguments arguments;
};

template <typename Gemm> class GemmBiasOperation : public GemmOperation<Gemm> {
public:
  using ElementA = typename Gemm::ElementA;
  using LayoutA = typename Gemm::LayoutA;
  using ElementB = typename Gemm::ElementB;
  using LayoutB = typename Gemm::LayoutB;
  using ElementC = typename Gemm::ElementC;
  using LayoutC = typename Gemm::LayoutC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  GemmBiasOperation(const char *kernel_name)
      : GemmOperation<Gemm>(kernel_name, OperationEnum::MatmulBias) {}

  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c, void *d) override {
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
    this->arguments = {problem_size,
                       {(ElementA *)a, layoutA},
                       {(ElementB *)b, layoutB},
                       {(ElementC *)c, LayoutC(0)},
                       {(ElementC *)d, layoutC},
                       {(ElementAccumulator)1, (ElementAccumulator)0},
                       /*split_k_slices=*/1};
  }
};
