#pragma once

#include "Operation.h"

template <typename Conv2d> class Conv2dOperation : public Operation {
public:
  Conv2dOperation(const char *kernel_name) : Operation(kernel_name) {}
};
