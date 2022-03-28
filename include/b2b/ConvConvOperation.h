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

  using ElementScaleBias = typename ConvConv::ElementScaleBias;
  using LayoutScaleBias = typename ConvConv::LayoutScaleBias;

  ConvConvOperation(const char *kernel_name) : Operation(kernel_name) {
    trait =
        Operation::OperationTrait{OperationEnum::Conv2d,
                                  EpilogueEnum::None,
                                  cutlass_type_to_dtype_v<ElementA>,
                                  cutlass_layout_to_layout_v<LayoutA>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
    trait1 =
        Operation::OperationTrait{OperationEnum::Conv2d,
                                  EpilogueEnum::None,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  virtual void
  SetArgument(int64_t N0, int64_t iH0, int64_t iW0, int64_t iC0, int64_t oH0,
              int64_t oW0, int64_t oC0, int64_t kH0, int64_t kW0,
              int64_t strideH0, int64_t strideW0, int64_t paddingH0,
              int64_t paddingW0, int64_t dilationH0, int64_t dilationW0,
              int64_t N1, int64_t iH1, int64_t iW1, int64_t iC1, int64_t oH1,
              int64_t oW1, int64_t oC1, int64_t kH1, int64_t kW1,
              int64_t strideH1, int64_t strideW1, int64_t paddingH1,
              int64_t paddingW1, int64_t dilationH1, int64_t dilationW1,
              void *input0, void *filter0, void *bias0, void *filter1,
              void *bias1, void *output1, int64_t /*split_k_slices*/,
              float alpha0, float beta0, float alpha1, float beta1) override {
    typename cutlass::conv::Conv2dProblemSize problem_size_0(
        {N0, iH0, iW0, iC0}, {oC0, kH0, kW0, iC0},
        {paddingH0, paddingH0, paddingW0, paddingW0}, {strideH0, strideW0},
        {dilationH0, dilationW0}, {N0, oH0, oW0, oC0},
        cutlass::conv::Mode::kCrossCorrelation);
    typename cutlass::conv::Conv2dProblemSize problem_size_1(
        {N1, iH1, iW1, iC1}, {oC1, kH1, kW1, iC1},
        {paddingH1, paddingH1, paddingW1, paddingW1}, {strideH1, strideW1},
        {dilationH1, dilationW1}, {N1, oH1, oW1, oC1},
        cutlass::conv::Mode::kCrossCorrelation);

    assert(cutlass_layout_to_layout_v<LayoutA> == LayoutEnum::NHWC);
    assert(cutlass_layout_to_layout_v<LayoutB> == LayoutEnum::NHWC);
    assert(cutlass_layout_to_layout_v<LayoutC> == LayoutEnum::NHWC);
    assert(N0 == N1 && oH0 == iH1 && oW0 == iW1 && oC0 == iC1);
    LayoutA layoutA0(iC0, iC0 * iW0, iC0 * iW0 * iH0);
    LayoutB layoutB0(iC0, iC0 * kW0, iC0 * kW0 * kH0);
    LayoutC layoutC0(oC0, oC0 * oW0, oC0 * oW0 * oH0);
    LayoutB layoutB1(iC1, iC1 * kW1, iC1 * kW1 * kH1);
    LayoutC layoutD1(oC1, oC1 * oW1, oC1 * oW1 * oH1);
    arguments = {problem_size_0,
                 problem_size_1,
                 {(ElementA *)input0, layoutA0},
                 {(ElementB *)filter0, layoutB0},
                 {(ElementC *)nullptr, layoutC0},
                 cutlass::TensorRef<ElementScaleBias, LayoutScaleBias>(),
                 cutlass::TensorRef<ElementScaleBias, LayoutScaleBias>(),
                 {(ElementB *)filter1, layoutB1},
                 {(ElementC *)nullptr, layoutD1},
                 {(ElementC *)output1, layoutD1},
                 {(ElementAccumulator)alpha0, (ElementAccumulator)beta0},
                 {(ElementAccumulator)alpha1, (ElementAccumulator)beta1},
                 cutlass::conv::SplitKMode::kSerial};
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

  virtual const OperationTrait *Trait() override { return &trait; }

  virtual const OperationTrait *Trait1() override { return &trait1; }

private:
  ConvConv conv2d;
  typename Operation::OperationTrait trait;
  typename Operation::OperationTrait trait1;
  typename ConvConv::Arguments arguments;
};
