#pragma once

#include <vector>

#include "Operation.h"

class Manifest {
public:
  std::vector<Operation *> kernels;

public:
  void reserve(size_t count) { kernels.reserve(count); }

  void append(Operation *op) { kernels.push_back(op); }

  ~Manifest() {
    for (auto &kernel : kernels) {
      delete kernel;
    }
  }
};
