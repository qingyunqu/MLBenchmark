#pragma once

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"

#include "Operation.h"
#include "cutlass_dtype.h"

template <typename Conv2d> class Conv2dOperation : public Operation {
public:
  using ElementA = typename Conv2d::ElementA;
  using LayoutA = typename Conv2d::LayoutA;
  using ElementB = typename Conv2d::ElementB;
  using LayoutB = typename Conv2d::LayoutB;
  using ElementC = typename Conv2d::ElementC;
  using LayoutC = typename Conv2d::LayoutC;
  using ElementAccumulator = typename Conv2d::ElementAccumulator;

  Conv2dOperation(const char *kernel_name) : Operation(kernel_name) {
    trait = {OperationEnum::Conv2d,
             cutlass_type_to_dtype_v<ElementA>,
             cutlass_layout_to_layout_v<LayoutA>,
             cutlass_type_to_dtype_v<ElementB>,
             cutlass_layout_to_layout_v<LayoutB>,
             cutlass_type_to_dtype_v<ElementC>,
             cutlass_layout_to_layout_v<LayoutC>,
             cutlass_type_to_dtype_v<ElementAccumulator>};
  }

  virtual void SetArgument(int64_t, int64_t, int64_t, void *, void *, void *) {
    assert(false && "should not reach this");
  }

  virtual bool Check() { return true; }

  virtual int64_t GetWorkspaceSize() { return -1; }

  virtual void Initialize(cudaStream_t stream, void *workspace) {}

  virtual void Run() {}

  virtual const OperationTrait &Trait() { return trait; }

private:
  Conv2d conv2d;
  typename Conv2d::Arguments arguments;
  typename Operation::OperationTrait trait;
};
