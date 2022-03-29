#include "convolution/CudnnConv.h"
#include "check.h"
#include "cuda_dtype.h"
#include "ops.h"

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <string>

const char *cudnn_math_type_to_str(cudnnMathType_t mathType) {
  switch (mathType) {
  case CUDNN_DEFAULT_MATH:
    return "CUDNN_DEFAULT_MATH";
  case CUDNN_TENSOR_OP_MATH:
    return "CUDNN_TENSOR_OP_MATH";
  case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
    return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
  case CUDNN_FMA_MATH:
    return "CUDNN_FMA_MATH";
  default:
    return "";
  }
}

const char *cudnn_algo_to_str(cudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
    return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
    return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
    return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
    return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
    return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
  default:
    return "";
  }
}

template <typename T, typename To, typename CompOn>
CudnnConv<T, To, CompOn>::CudnnConv(const std::string &layout, int64_t N,
                                    int64_t iC, int64_t iH, int64_t iW,
                                    int64_t oC, int64_t kH, int64_t kW,
                                    int64_t oH, int64_t oW, int64_t strideH,
                                    int64_t strideW, int64_t paddingH,
                                    int64_t paddingW, int64_t dilateH,
                                    int64_t dilateW, cudnnHandle_t handle)
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
  CUDNNCHECK(cudnnSetConvolutionMathType(convolution_descriptor,
                                         CUDNN_TENSOR_OP_MATH));
  CUDNNCHECK(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_h=*/paddingH,
                                             /*pad_w=*/paddingW,
                                             /*u=*/strideH,
                                             /*v=*/strideW,
                                             /*dilation_h=*/dilateH,
                                             /*dilation_w=*/dilateW,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/compute_dtype));

  int returnedAlgoCount;
  // cudnnConvolutionFwdAlgoPerf_t results[10];
  CUDNNCHECK(cudnnFindConvolutionForwardAlgorithm(
      handle, input_descriptor, filter_descriptor, convolution_descriptor,
      output_descriptor,
      /*requestedAlgoCount=*/1, &returnedAlgoCount, &perf));
  assert(returnedAlgoCount == 1);
  // printf("returnedAlgoCount: %d\n", returnedAlgoCount);
  // for (int i = 0; i < returnedAlgoCount; i++) {
  //   printf("algo: %s\n", cudnn_algo_to_str(results[i].algo));
  //   printf("time: %f ms\n", results[i].time);
  //   printf("mathType: %s\n\n", cudnn_math_type_to_str(results[i].mathType));
  // }
  // perf = results[0];

  CUDACHECK(cudaMalloc(&workspace, perf.memory));
}

template <typename T, typename To, typename CompOn>
void CudnnConv<T, To, CompOn>::Run(const T *input, const T *filter,
                                   To *output) {
  float alpha = 1.f, beta = 0.f;
  CUDNNCHECK(cudnnConvolutionForward(
      handle, &alpha, input_descriptor, input, filter_descriptor, filter,
      convolution_descriptor, perf.algo, workspace, perf.memory, &beta,
      output_descriptor, output));
}

template <typename T, typename To, typename CompOn>
CudnnConv<T, To, CompOn>::~CudnnConv() {
  CUDACHECK(cudaFree(workspace));
  CUDNNCHECK(cudnnDestroyTensorDescriptor(input_descriptor));
  CUDNNCHECK(cudnnDestroyTensorDescriptor(output_descriptor));
  CUDNNCHECK(cudnnDestroyFilterDescriptor(filter_descriptor));
  CUDNNCHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
}

// instantiate
template class CudnnConv<float, float, float>;
template class CudnnConv<__half, __half, float>;
template class CudnnConv<__half, __half, __half>;
