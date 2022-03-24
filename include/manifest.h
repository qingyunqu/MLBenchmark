#pragma once

#include <vector>

#include "Operation.h"
#include "dtype.h"

class Manifest {
private:
  std::vector<Operation *> kernels;

public:
  void reserve(size_t count) { kernels.reserve(count); }

  void append(Operation *op) { kernels.push_back(op); }

  template <typename ElementInputA, typename ElementInputB,
            typename ElementOutput, typename ElementAccumulator>
  void profile(int64_t m, int64_t n, int64_t k, LayoutEnum layout_a,
               LayoutEnum layout_b, LayoutEnum layout_c);

  ~Manifest() {
    for (auto &kernel : kernels) {
      delete kernel;
    }
  }
};
