#pragma once

#include <cudnn.h>
#include <string>

#include "ops.h"

template <typename T, typename To, typename CompOn>
class CudnnConv : public Conv<T, To> {
public:
  CudnnConv(const std::string &layout, int64_t N, int64_t iC, int64_t iH,
            int64_t iW, int64_t oC, int64_t kH, int64_t kW, int64_t oH,
            int64_t oW, int64_t strideH, int64_t strideW, int64_t paddingH,
            int64_t paddingW, int64_t dilateH, int64_t dilateW,
            cudnnHandle_t handle);
  virtual bool Check() override { return true; }
  virtual void Run(const T *input, const T *filter, To *output) override;
  virtual ~CudnnConv();

private:
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnFilterDescriptor_t filter_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnConvolutionFwdAlgoPerf_t perf;
  void *workspace = nullptr;
};