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

  template <typename TA, typename TB, typename TC, typename CompOn>
  void profile_gemm(int64_t m, int64_t n, int64_t k, LayoutEnum layout_a,
                    LayoutEnum layout_b, LayoutEnum layout_c);

  template <typename TA, typename TB, typename TC, typename CompOn>
  void profile_conv2d(int64_t N, int64_t iH, int64_t iW, int64_t iC, int64_t oH,
                      int64_t oW, int64_t oC, int64_t kH, int64_t kW,
                      int64_t strideH, int64_t strideW, int64_t paddingH,
                      int64_t paddingW, int64_t dilationH = 1,
                      int64_t dilationW = 1);

  ~Manifest() {
    for (auto &kernel : kernels) {
      delete kernel;
    }
  }
};
