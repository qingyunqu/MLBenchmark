#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <string>

#include "../check.h"
#include "../cuda_dtype.h"
#include "../ops.h"

template <typename T, typename To = T, typename CompOn = float>
class CudnnConv : public Conv<T, To> {
public:
  CudnnConv(const std::string &layout, int64_t N, int64_t iC, int64_t iH,
            int64_t iW, int64_t oC, int64_t kH, int64_t kW, int64_t oH,
            int64_t oW, int64_t strideH, int64_t strideW, int64_t paddingH,
            int64_t paddingW, int64_t dilateH, int64_t dilateW,
            cudnnHandle_t handle)
      : Conv<T, To>(layout, N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW,
                    paddingH, paddingW, dilateH, dilateW),
        handle(handle) {
    assert(layout == "NCHW" || layout == "NHWC");
    cudnnTensorFormat_t format;
    if (layout == "NCHW") {
      format = CUDNN_TENSOR_NCHW;
    } else if (layout == "NHWC") {
      format = CUDNN_TENSOR_NHWC;
    }

    auto input_dtype = ctype_to_cudnn_dtype<T>::value;
    auto output_dtype = ctype_to_cudnn_dtype<To>::value;
    auto compute_dtype = ctype_to_cudnn_dtype<CompOn>::value;
    CUDNNCHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNNCHECK(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/format,
                                          /*dataType=*/input_dtype,
                                          /*batch_size=*/N,
                                          /*channels=*/iC,
                                          /*image_height=*/iH,
                                          /*image_width=*/iW));
    CUDNNCHECK(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNNCHECK(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/format,
                                          /*dataType=*/output_dtype,
                                          /*batch_size=*/N,
                                          /*channels=*/oC,
                                          /*image_height=*/oH,
                                          /*image_width=*/oW));
    CUDNNCHECK(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNNCHECK(cudnnSetFilter4dDescriptor(filter_descriptor,
                                          /*dataType=*/input_dtype,
                                          /*format=*/format,
                                          /*out_channels=*/oC,
                                          /*in_channels=*/iC,
                                          /*kernel_height=*/kH,
                                          /*kernel_width=*/kW));
    CUDNNCHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CUDNNCHECK(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                               /*pad_h=*/paddingH,
                                               /*pad_w=*/paddingW,
                                               /*u=*/strideH,
                                               /*v=*/strideW,
                                               /*dilation_h=*/dilateH,
                                               /*dilation_w=*/dilateW,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/compute_dtype));

    int returnedAlgoCount = 0;
    CUDNNCHECK(cudnnFindConvolutionForwardAlgorithm(
        handle, input_descriptor, filter_descriptor, convolution_descriptor,
        output_descriptor,
        /*requestedAlgoCount=*/1, &returnedAlgoCount, &perf));
    assert(returnedAlgoCount == 1);

    CUDACHECK(cudaMalloc(&workspace, perf.memory));
  }
  virtual bool Check() override { return true; }
  virtual void Run(const T *input, const T *filter, To *output) override {
    float alpha = 1.f, beta = 0.f;
    CUDNNCHECK(cudnnConvolutionForward(
        handle, &alpha, input_descriptor, input, filter_descriptor, filter,
        convolution_descriptor, perf.algo, workspace, perf.memory, &beta,
        output_descriptor, output));
  }
  virtual ~CudnnConv() {
    CUDACHECK(cudaFree(workspace));
    CUDNNCHECK(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNNCHECK(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNNCHECK(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNNCHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
  }

private:
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnFilterDescriptor_t filter_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnConvolutionFwdAlgoPerf_t perf;
  void *workspace = nullptr;
};