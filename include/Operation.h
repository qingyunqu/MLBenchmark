#pragma once

#include "dtype.h"
#include <cuda_runtime.h>

class Operation {
public:
  struct OperationTrait {
    DTypeEnum element_a;
    LayoutEnum layout_a;
    DTypeEnum element_b;
    LayoutEnum layout_b;
    DTypeEnum element_c;
    LayoutEnum layout_c;
    DTypeEnum accumulator;
    bool operator!=(const OperationTrait &trait) const {
      return !(element_a == trait.element_a && element_b == trait.element_b &&
               element_c == trait.element_c && layout_a == trait.layout_a &&
               layout_b == trait.layout_b && layout_c == trait.layout_c &&
               accumulator == trait.accumulator);
    }
  };

  Operation(const char *kernel_name) : kernel_name(kernel_name) {}
  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c) = 0;
  virtual bool Check() = 0;
  virtual void Initialize(cudaStream_t) = 0;
  virtual void Run() = 0;
  const char *Name() { return kernel_name; }
  virtual const OperationTrait &Trait() = 0;

private:
  const char *kernel_name;
};
