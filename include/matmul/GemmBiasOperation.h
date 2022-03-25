#pragma once

#include "matmul/GemmOperation.h"

template <typename Gemm> class GemmBiasOperation : public GemmOperation<Gemm> {
public:
  using ElementA = typename Gemm::ElementA;
  using LayoutA = typename Gemm::LayoutA;
  using ElementB = typename Gemm::ElementB;
  using LayoutB = typename Gemm::LayoutB;
  using ElementC = typename Gemm::ElementC;
  using LayoutC = typename Gemm::LayoutC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  GemmBiasOperation(const char *kernel_name, EpilogueEnum epilogue_enum)
      : GemmOperation<Gemm>(kernel_name, OperationEnum::MatmulBias,
                            epilogue_enum) {}

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
