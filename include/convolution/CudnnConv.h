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
  virtual void AllocWorkspace() override;
  virtual void SetArgument(T *_input, T *_filter, To *_output) override {
    input = _input;
    filter = _filter;
    output = _output;
  }
  virtual void Run() override;
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
  T *input = nullptr, *filter = nullptr;
  To *output = nullptr;
};

template <typename T, typename To, typename CompOn>
class CudnnConvBias : public CudnnConv<T, To, CompOn> {
public:
  CudnnConvBias(const std::string &layout, int64_t N, int64_t iC, int64_t iH,
                int64_t iW, int64_t oC, int64_t kH, int64_t kW, int64_t oH,
                int64_t oW, int64_t strideH, int64_t strideW, int64_t paddingH,
                int64_t paddingW, int64_t dilateH, int64_t dilateW,
                cudnnHandle_t handle, EpilogueEnum epilogue);

  virtual void SetArgument(T *_input, T *_filter, To *_bias,
                           To *_output) override {
    this->input = _input;
    this->filter = _filter;
    this->bias = _bias;
    this->output = _output;
  }
  virtual void Run() override;

  virtual ~CudnnConvBias();

private:
  cudnnTensorDescriptor_t bias_descriptor;
  cudnnActivationDescriptor_t act_descriptor;
  To *bias = nullptr;
};

template <typename T> class CudnnActivation : public Op<T> {
public:
  CudnnActivation(const std::vector<int64_t> &shape, EpilogueEnum epilogue,
                  cudnnHandle_t handle);

  virtual bool Check() { return true; }

  virtual void SetArgument(T *_input) override { this->input = _input; }
  virtual void Run() override;

  virtual ~CudnnActivation();

private:
  EpilogueEnum epilogue;
  T *input = nullptr;
  cudnnHandle_t(handle);
  cudnnTensorDescriptor_t input_descriptor;
  cudnnActivationDescriptor_t act_descriptor;
};

template <typename T>
void CudnnActivate(cudnnHandle_t handle, T *input,
                   const std::vector<int64_t> &shape, EpilogueEnum epilogue);
