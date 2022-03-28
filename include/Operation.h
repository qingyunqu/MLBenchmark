#pragma once

#include "dtype.h"
#include <cassert>
#include <cuda_runtime.h>

class Operation {
public:
  struct OperationTrait {
    OperationEnum operation;
    EpilogueEnum epilogue;
    DTypeEnum element_a;
    LayoutEnum layout_a;
    DTypeEnum element_b;
    LayoutEnum layout_b;
    DTypeEnum element_c;
    LayoutEnum layout_c;
    DTypeEnum accumulator;

    OperationTrait() : operation(OperationEnum::Invalid) {}
    OperationTrait(OperationEnum operation, EpilogueEnum epilogue,
                   DTypeEnum element_a, LayoutEnum layout_a,
                   DTypeEnum element_b, LayoutEnum layout_b,
                   DTypeEnum element_c, LayoutEnum layout_c,
                   DTypeEnum accumulator)
        : operation(operation), epilogue(epilogue), element_a(element_a),
          layout_a(layout_a), element_b(element_b), layout_b(layout_b),
          element_c(element_c), layout_c(layout_c), accumulator(accumulator) {}
    OperationTrait(const OperationTrait &trait)
        : operation(trait.operation), epilogue(trait.epilogue),
          element_a(trait.element_a), layout_a(trait.layout_a),
          element_b(trait.element_b), layout_b(trait.layout_b),
          element_c(trait.element_c), layout_c(trait.layout_c),
          accumulator(trait.accumulator) {}

    bool operator!=(const OperationTrait &trait) const {
      if (operation == OperationEnum::Invalid ||
          trait.operation == OperationEnum::Invalid) {
        return true;
      }
      return !(operation == trait.operation && epilogue == trait.epilogue &&
               element_a == trait.element_a && element_b == trait.element_b &&
               element_c == trait.element_c && layout_a == trait.layout_a &&
               layout_b == trait.layout_b && layout_c == trait.layout_c &&
               accumulator == trait.accumulator);
    }

    bool operator==(const OperationTrait &trait) const {
      if (operation == OperationEnum::Invalid ||
          trait.operation == OperationEnum::Invalid) {
        return false;
      }
      return operation == trait.operation && epilogue == trait.epilogue &&
             element_a == trait.element_a && element_b == trait.element_b &&
             element_c == trait.element_c && layout_a == trait.layout_a &&
             layout_b == trait.layout_b && layout_c == trait.layout_c &&
             accumulator == trait.accumulator;
    }
  };

  Operation(const char *kernel_name) : kernel_name(kernel_name) {}
  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c, void *d, int64_t split_k_slices,
                           float alpha, float beta) {
    assert(false && "should not reach this");
  }
  virtual void SetArgument(int64_t N, int64_t iH, int64_t iW, int64_t iC,
                           int64_t oH, int64_t oW, int64_t oC, int64_t kH,
                           int64_t kW, int64_t strideH, int64_t strideW,
                           int64_t paddingH, int64_t paddingW,
                           int64_t dilationH, int64_t dilationW, void *input,
                           void *filter, void *bias, void *output,
                           int64_t split_k_slices, float alpha, float beta) {
    assert(false && "should not reach this");
  }
  virtual bool Check() = 0;
  virtual int64_t GetWorkspaceSize() = 0;
  virtual void Initialize(cudaStream_t stream, void *workspace) = 0;
  virtual void Run() = 0;
  virtual const OperationTrait *Trait() = 0;
  virtual const OperationTrait *Trait1() { return nullptr; }
  const char *Name() { return kernel_name; }

private:
  const char *kernel_name;
};
