#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "../check.h"

inline const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

template <typename T, typename CompOn = float> class CublasMatmul {
public:
  void Run(const T *a_val, const T *b_val, T *c_val, int64_t m, int64_t n,
           int64_t k, bool lhs_transpose, bool rhs_transpose,
           bool output_transpose, cublasHandle_t handle);
};

template <>
void CublasMatmul<float>::Run(const float *a_val, const float *b_val,
                              float *c_val, int64_t m, int64_t n, int64_t k,
                              bool lhs_transpose, bool rhs_transpose,
                              bool output_transpose, cublasHandle_t handle) {
  float alpha = 1.0f, beta = 0.0f;
  if (!output_transpose) {
    if (!lhs_transpose && !rhs_transpose) {
      // CT = (AB)T = BT @ AT
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              b_val, n, a_val, k, &beta, c_val, n));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                              b_val, k, a_val, k, &beta, c_val, n));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha,
                              b_val, n, a_val, m, &beta, c_val, n));
    } else {
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha,
                              b_val, k, a_val, m, &beta, c_val, n));
    }
  } else {
    if (!lhs_transpose && !rhs_transpose) {
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha,
                              a_val, k, b_val, n, &beta, c_val, m));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                              a_val, k, b_val, k, &beta, c_val, m));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                              a_val, m, b_val, n, &beta, c_val, m));
    } else {
      CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                              a_val, m, b_val, k, &beta, c_val, m));
    }
  }
}

template <>
void CublasMatmul<__half, float>::Run(const __half *a_val, const __half *b_val,
                                      __half *c_val, int64_t m, int64_t n,
                                      int64_t k, bool lhs_transpose,
                                      bool rhs_transpose, bool output_transpose,
                                      cublasHandle_t handle) {
  float alpha = 1.0f, beta = 0.0f;
  // compute on fp32
  if (!output_transpose) {
    if (!lhs_transpose && !rhs_transpose) {
      // CT = (AB)T = BT @ AT
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, b_val, CUDA_R_16F, n, a_val, CUDA_R_16F,
                                k, &beta, c_val, CUDA_R_16F, n));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                &alpha, b_val, CUDA_R_16F, k, a_val, CUDA_R_16F,
                                k, &beta, c_val, CUDA_R_16F, n));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
                                &alpha, b_val, CUDA_R_16F, n, a_val, CUDA_R_16F,
                                m, &beta, c_val, CUDA_R_16F, n));
    } else {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k,
                                &alpha, b_val, CUDA_R_16F, k, a_val, CUDA_R_16F,
                                m, &beta, c_val, CUDA_R_16F, n));
    }
  } else {
    if (!lhs_transpose && !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                &alpha, a_val, CUDA_R_16F, k, b_val, CUDA_R_16F,
                                n, &beta, c_val, CUDA_R_16F, m));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                &alpha, a_val, CUDA_R_16F, k, b_val, CUDA_R_16F,
                                k, &beta, c_val, CUDA_R_16F, m));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                &alpha, a_val, CUDA_R_16F, m, b_val, CUDA_R_16F,
                                n, &beta, c_val, CUDA_R_16F, m));
    } else {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                &alpha, a_val, CUDA_R_16F, m, b_val, CUDA_R_16F,
                                k, &beta, c_val, CUDA_R_16F, m));
    }
  }
}

template <>
void CublasMatmul<__half, __half>::Run(const __half *a_val, const __half *b_val,
                                       __half *c_val, int64_t m, int64_t n,
                                       int64_t k, bool lhs_transpose,
                                       bool rhs_transpose,
                                       bool output_transpose,
                                       cublasHandle_t handle) {
  __half alpha = static_cast<__half>(1.0f);
  __half beta = static_cast<__half>(0.0f);
  if (!output_transpose) {
    if (!lhs_transpose && !rhs_transpose) {
      // CT = (AB)T = BT @ AT
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              b_val, n, a_val, k, &beta, c_val, n));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                              b_val, k, a_val, k, &beta, c_val, n));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha,
                              b_val, n, a_val, m, &beta, c_val, n));
    } else {
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha,
                              b_val, k, a_val, m, &beta, c_val, n));
    }
  } else {
    if (!lhs_transpose && !rhs_transpose) {
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha,
                              a_val, k, b_val, n, &beta, c_val, m));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                              a_val, k, b_val, k, &beta, c_val, m));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                              a_val, m, b_val, n, &beta, c_val, m));
    } else {
      CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                              a_val, m, b_val, k, &beta, c_val, m));
    }
  }
}

template <>
void CublasMatmul<__nv_bfloat16, float>::Run(
    const __nv_bfloat16 *a_val, const __nv_bfloat16 *b_val,
    __nv_bfloat16 *c_val, int64_t m, int64_t n, int64_t k, bool lhs_transpose,
    bool rhs_transpose, bool output_transpose, cublasHandle_t handle) {
  float alpha = 1.0f, beta = 0.0f;
  // compute on fp32
  if (!output_transpose) {
    if (!lhs_transpose && !rhs_transpose) {
      // CT = (AB)T = BT @ AT
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, b_val, CUDA_R_16BF, n, a_val,
                                CUDA_R_16BF, k, &beta, c_val, CUDA_R_16BF, n));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                &alpha, b_val, CUDA_R_16BF, k, a_val,
                                CUDA_R_16BF, k, &beta, c_val, CUDA_R_16BF, n));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
                                &alpha, b_val, CUDA_R_16BF, n, a_val,
                                CUDA_R_16BF, m, &beta, c_val, CUDA_R_16BF, n));
    } else {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k,
                                &alpha, b_val, CUDA_R_16BF, k, a_val,
                                CUDA_R_16BF, m, &beta, c_val, CUDA_R_16BF, n));
    }
  } else {
    if (!lhs_transpose && !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                &alpha, a_val, CUDA_R_16BF, k, b_val,
                                CUDA_R_16BF, n, &beta, c_val, CUDA_R_16BF, m));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                &alpha, a_val, CUDA_R_16BF, k, b_val,
                                CUDA_R_16BF, k, &beta, c_val, CUDA_R_16BF, m));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                &alpha, a_val, CUDA_R_16BF, m, b_val,
                                CUDA_R_16BF, n, &beta, c_val, CUDA_R_16BF, m));
    } else {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                &alpha, a_val, CUDA_R_16BF, m, b_val,
                                CUDA_R_16BF, k, &beta, c_val, CUDA_R_16BF, m));
    }
  }
}