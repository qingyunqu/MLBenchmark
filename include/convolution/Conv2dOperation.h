#pragma once

#include "../manifest.h"

template <typename Conv2d> class Conv2dOperation : public Operation {
public:
  Conv2dOperation(const char *kernel_name) : Operation(kernel_name) {}
};
