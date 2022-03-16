#pragma once

#include <cuda_runtime.h>

#include "../check.h"
#include "../ops.h"

template <typename T, typename CompOn = float>
class CutlassMatmul : public Matmul<T> {
public:
  using Matmul<T>::Matmul;
};