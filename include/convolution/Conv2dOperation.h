#pragma once

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"

#include "Operation.h"
#include "check.h"
#include "cutlass_dtype.h"

#include <iostream>

template <typename Conv2d> class Conv2dOperation : public Operation {
public:
  using ElementA = typename Conv2d::ElementA;
  using LayoutA = typename Conv2d::LayoutA;
  using ElementB = typename Conv2d::ElementB;
  using LayoutB = typename Conv2d::LayoutB;
  using ElementC = typename Conv2d::ElementC;
  using LayoutC = typename Conv2d::LayoutC;
  using ElementAccumulator = typename Conv2d::ElementAccumulator;

  Conv2dOperation(const char *kernel_name,
                  EpilogueEnum epilogue_enum = EpilogueEnum::None)
      : Operation(kernel_name) {
    trait =
        Operation::OperationTrait{OperationEnum::Conv2d,
                                  epilogue_enum,
                                  cutlass_type_to_dtype_v<ElementA>,
                                  cutlass_layout_to_layout_v<LayoutA>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  Conv2dOperation(const char *kernel_name, OperationEnum op_enum,
                  EpilogueEnum epilogue_enum)
      : Operation(kernel_name) {
    trait =
        Operation::OperationTrait{op_enum,
                                  epilogue_enum,
                                  cutlass_type_to_dtype_v<ElementA>,
                                  cutlass_layout_to_layout_v<LayoutA>,
                                  cutlass_type_to_dtype_v<ElementB>,
                                  cutlass_layout_to_layout_v<LayoutB>,
                                  cutlass_type_to_dtype_v<ElementC>,
                                  cutlass_layout_to_layout_v<LayoutC>,
                                  cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  virtual void SetArgument(int64_t N, int64_t iH, int64_t iW, int64_t iC,
                           int64_t oH, int64_t oW, int64_t oC, int64_t kH,
                           int64_t kW, int64_t strideH, int64_t strideW,
                           int64_t paddingH, int64_t paddingW,
                           int64_t dilationH, int64_t dilationW, void *input,
                           void *filter, void *bias, void *output) override {
    cutlass::Tensor4DCoord input_size(N, iH, iW, iC);
    cutlass::Tensor4DCoord filter_size(oC, kH, kW, iC);
    cutlass::Tensor4DCoord output_size(N, oH, oW, oC);
    typename cutlass::conv::Conv2dProblemSize problem_size(
        input_size, filter_size, {paddingH, paddingH, paddingW, paddingW},
        {strideH, strideW}, {dilationH, dilationW}, output_size,
        cutlass::conv::Mode::kCrossCorrelation,
        /*split_k_slices*/ 1);

    assert(cutlass_layout_to_layout_v<LayoutA> == LayoutEnum::NHWC);
    assert(cutlass_layout_to_layout_v<LayoutB> == LayoutEnum::NHWC);
    assert(cutlass_layout_to_layout_v<LayoutC> == LayoutEnum::NHWC);
    LayoutA layoutA(iC, iC * iW, iC * iW * iH);
    LayoutB layoutB(iC, iC * kW, iC * kW * kH);
    LayoutC layoutC(oC, oC * oW, oC * oW * oH);
    arguments = {problem_size,
                 {(ElementA *)input, layoutA},
                 {(ElementB *)filter, layoutB},
                 {(ElementC *)bias, layoutC},
                 {(ElementC *)output, layoutC},
                 {(ElementAccumulator)1, (ElementAccumulator)0}};
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

private:
  Conv2d conv2d;
  typename Operation::OperationTrait trait;

protected:
  typename Conv2d::Arguments arguments;
};
