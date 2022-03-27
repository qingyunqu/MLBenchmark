#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "Operation.h"
#include "check.h"
#include "cutlass_dtype.h"
#include <iostream>

template <typename ConvConv> class ConvConvOperation : public Operation {
public:
  using ElementA = typename ConvConv::ElementA;
  using LayoutA = typename ConvConv::LayoutA;
  using ElementB = typename ConvConv::ElementB;
  using LayoutB = typename ConvConv::LayoutB;
  using ElementC = typename ConvConv::ElementC;
  using LayoutC = typename ConvConv::LayoutC;
  using ElementAccumulator = typename ConvConv::ElementAccumulator;

  ConvConvOperation(const char *kernel_name) : Operation(kernel_name) {}

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
    // arguments = {problem_size,
    //              {(ElementA *)input, layoutA},
    //              {(ElementB *)filter, layoutB},
    //              {(ElementC *)bias, layoutC},
    //              {(ElementC *)output, layoutC},
    //              {(ElementAccumulator)alpha, (ElementAccumulator)beta}};
  }

  virtual bool Check() override {
    return conv2d.can_implement(arguments) == cutlass::Status::kSuccess;
  }

  virtual int64_t GetWorkspaceSize() override {
    return conv2d.get_workspace_size(arguments);
  }

  virtual void Initialize(cudaStream_t stream, void *workspace) override {
    CUTLASS_CHECK(conv2d.initialize(arguments, workspace, stream));
  }

  virtual void Run() override { CUTLASS_CHECK(conv2d()); }

  virtual const OperationTrait &Trait() override { return trait; }

  virtual const OperationTrait &Trait1() override { return trait1; }

private:
  ConvConv conv2d;
  typename Operation::OperationTrait trait;
  typename Operation::OperationTrait trait1;
  typename ConvConv::Arguments arguments;
};
