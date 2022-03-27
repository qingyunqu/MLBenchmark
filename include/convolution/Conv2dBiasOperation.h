#pragma once

#include "convolution/Conv2dOperation.h"

template <typename Conv2d>
class Conv2dBiasOperation : public Conv2dOperation<Conv2d> {
public:
  using ElementA = typename Conv2d::ElementA;
  using LayoutA = typename Conv2d::LayoutA;
  using ElementB = typename Conv2d::ElementB;
  using LayoutB = typename Conv2d::LayoutB;
  using ElementC = typename Conv2d::ElementC;
  using LayoutC = typename Conv2d::LayoutC;
  using ElementAccumulator = typename Conv2d::ElementAccumulator;

  Conv2dBiasOperation(const char *kernel_name, EpilogueEnum epilogue_enum)
      : Conv2dOperation<Conv2d>(kernel_name, OperationEnum::Conv2dBias,
                                epilogue_enum) {}

  virtual void SetArgument(int64_t N, int64_t iH, int64_t iW, int64_t iC,
                           int64_t oH, int64_t oW, int64_t oC, int64_t kH,
                           int64_t kW, int64_t strideH, int64_t strideW,
                           int64_t paddingH, int64_t paddingW,
                           int64_t dilationH, int64_t dilationW, void *input,
                           void *filter, void *bias, void *output,
                           int64_t split_k_slices, float alpha,
                           float beta) override {
    cutlass::Tensor4DCoord input_size(N, iH, iW, iC);
    cutlass::Tensor4DCoord filter_size(oC, kH, kW, iC);
    cutlass::Tensor4DCoord output_size(N, oH, oW, oC);
    typename cutlass::conv::Conv2dProblemSize problem_size(
        input_size, filter_size, {paddingH, paddingH, paddingW, paddingW},
        {strideH, strideW}, {dilationH, dilationW}, output_size,
        cutlass::conv::Mode::kCrossCorrelation, split_k_slices);

    assert(cutlass_layout_to_layout_v<LayoutA> == LayoutEnum::NHWC);
    assert(cutlass_layout_to_layout_v<LayoutB> == LayoutEnum::NHWC);
    assert(cutlass_layout_to_layout_v<LayoutC> == LayoutEnum::NHWC);
    LayoutA layoutA(iC, iC * iW, iC * iW * iH);
    LayoutB layoutB(iC, iC * kW, iC * kW * kH);
    LayoutC layoutC(oC, oC * oW, oC * oW * oH);
    this->arguments = {problem_size,
                       {(ElementA *)input, layoutA},
                       {(ElementB *)filter, layoutB},
                       {(ElementC *)bias, LayoutC()},
                       {(ElementC *)output, layoutC},
                       {(ElementAccumulator)alpha, (ElementAccumulator)beta}};
  }
};