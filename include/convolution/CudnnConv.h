#pragma once

#include <cudnn.h>
#include <string>
#include <vector>

#include "dtype.h"
#include "ops.h"

template <typename T, typename To, typename CompOn>
class CudnnConv : public Op<T, To> {
public:
  CudnnConv(const std::string &layout, int64_t N, int64_t iC, int64_t iH,
            int64_t iW, int64_t oC, int64_t kH, int64_t kW, int64_t oH,
            int64_t oW, int64_t strideH, int64_t strideW, int64_t paddingH,
            int64_t paddingW, int64_t dilateH, int64_t dilateW,
            cudnnHandle_t handle);
  virtual bool Check() override { return true; }
  virtual void Initialize() override;
  virtual void Run(T *input, T *filter, To *output) override;
  virtual ~CudnnConv();

protected:
  cudnnHandle_t handle;
  cudnnTensorFormat_t format;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnFilterDescriptor_t filter_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnConvolutionFwdAlgoPerf_t perf;
  void *workspace = nullptr;
};

template <typename T, typename To, typename CompOn>
class CudnnConvBias : public CudnnConv<T, To, CompOn> {
public:
  CudnnConvBias(const std::string &layout, int64_t N, int64_t iC, int64_t iH,
                int64_t iW, int64_t oC, int64_t kH, int64_t kW, int64_t oH,
                int64_t oW, int64_t strideH, int64_t strideW, int64_t paddingH,
                int64_t paddingW, int64_t dilateH, int64_t dilateW,
                cudnnHandle_t handle, EpilogueEnum epilogue);

  virtual void Run(T *input, T *filter, To *bias, To *output) override;

  virtual ~CudnnConvBias();

private:
  cudnnTensorDescriptor_t bias_descriptor;
  cudnnActivationDescriptor_t act_descriptor;
  EpilogueEnum epilogue;
};

template <typename T>
void CudnnActivate(cudnnHandle_t handle, T *input,
                   const std::vector<int64_t> &shape, EpilogueEnum epilogue);
