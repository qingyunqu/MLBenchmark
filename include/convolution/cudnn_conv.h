#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../check.h"
#include "../ops.h"

template <typename T, typename CompOn = float>
class CudnnConv : public Conv<T> {
public:
  CudnnConv(const string &layout, int64_t N, int64_t iC, int64_t iH, int64_t iW,
            int64_t oC, int64_t kH, int64_t kW, int64_t oH, int64_t oW,
            int64_t strideH, int64_t strideW, int64_t paddingH,
            int64_t paddingW, int64_t dilateH, int64_t dilateW,
            cudnnHandle_t handle)
      : Conv<T>(layout, N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW,
                paddingH, paddingW, dilateH, dilateW),
        handle(handle) {}
  virtual void Run(const T *input, const T *filter, T *output) override;
  virtual ~CudnnConv() = default;

private:
  cudnnHandle_t handle;
}