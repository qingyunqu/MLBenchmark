#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

// cublas
template <typename T> struct ctype_to_cublas_dtype {};
template <> struct ctype_to_cublas_dtype<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};
template <> struct ctype_to_cublas_dtype<__half> {
  static constexpr cudaDataType_t value = CUDA_R_16F;
};
template <> struct ctype_to_cublas_dtype<__nv_bfloat16> {
  static constexpr cudaDataType_t value = CUDA_R_16BF;
};

// cudnn
template <typename T> struct ctype_to_cudnn_dtype {};
template <> struct ctype_to_cudnn_dtype<float> {
  static constexpr cudnnDataType_t value = CUDNN_DATA_FLOAT;
};
template <> struct ctype_to_cudnn_dtype<__half> {
  static constexpr cudnnDataType_t value = CUDNN_DATA_HALF;
};
