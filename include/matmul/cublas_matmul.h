#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "../check.h"
#include "../ops.h"

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

template <typename T, typename To, typename CompOn>
class CublasMatmul : public Matmul<T, To> {
public:
  CublasMatmul(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
               bool rhs_transpose, bool output_transpose, cublasHandle_t handle)
      : Matmul<T, To>(m, n, k, lhs_transpose, rhs_transpose, output_transpose),
        handle(handle) {}
  virtual bool Check() override { return true; }
  virtual void Run(const T *a_val, const T *b_val, To *c_val) override;
  virtual ~CublasMatmul() = default;

private:
  cublasHandle_t handle;
};

//===----------------------------------------------------------------------===//
// cublasSgemm
//===----------------------------------------------------------------------===//
template <>
void CublasMatmul<float, float, float>::Run(const float *a_val,
                                            const float *b_val, float *c_val) {
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

//===----------------------------------------------------------------------===//
// cublasSgemmEx
//===----------------------------------------------------------------------===//

// outside function for cublasSgemmEx
template <typename T, typename To>
inline void run_SgemmEx(const T *a_val, const T *b_val, To *c_val,
                        cudaDataType_t input_dtype, cudaDataType_t output_dtype,
                        int64_t m, int64_t n, int64_t k, bool lhs_transpose,
                        bool rhs_transpose, bool output_transpose,
                        cublasHandle_t handle) {
  float alpha = 1.0f, beta = 0.0f;
  // compute on fp32
  if (!output_transpose) {
    if (!lhs_transpose && !rhs_transpose) {
      // CT = (AB)T = BT @ AT
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, b_val, input_dtype, n, a_val,
                                input_dtype, k, &beta, c_val, output_dtype, n));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                &alpha, b_val, input_dtype, k, a_val,
                                input_dtype, k, &beta, c_val, output_dtype, n));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
                                &alpha, b_val, input_dtype, n, a_val,
                                input_dtype, m, &beta, c_val, output_dtype, n));
    } else {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k,
                                &alpha, b_val, input_dtype, k, a_val,
                                input_dtype, m, &beta, c_val, output_dtype, n));
    }
  } else {
    if (!lhs_transpose && !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                &alpha, a_val, input_dtype, k, b_val,
                                input_dtype, n, &beta, c_val, output_dtype, m));
    } else if (!lhs_transpose & rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                &alpha, a_val, input_dtype, k, b_val,
                                input_dtype, k, &beta, c_val, output_dtype, m));
    } else if (lhs_transpose & !rhs_transpose) {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                &alpha, a_val, input_dtype, m, b_val,
                                input_dtype, n, &beta, c_val, output_dtype, m));
    } else {
      CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                &alpha, a_val, input_dtype, m, b_val,
                                input_dtype, k, &beta, c_val, output_dtype, m));
    }
  }
}

template <>
void CublasMatmul<__half, __half, float>::Run(const __half *a_val,
                                              const __half *b_val,
                                              __half *c_val) {
  auto input_dtype = ctype_to_cublas_dtype<__half>::value;
  auto output_dtype = ctype_to_cublas_dtype<__half>::value;
  run_SgemmEx(a_val, b_val, c_val, input_dtype, output_dtype, m, n, k,
              lhs_transpose, rhs_transpose, output_transpose, handle);
}

template <>
void CublasMatmul<__half, float, float>::Run(const __half *a_val,
                                             const __half *b_val,
                                             float *c_val) {
  auto input_dtype = ctype_to_cublas_dtype<__half>::value;
  auto output_dtype = ctype_to_cublas_dtype<float>::value;
  run_SgemmEx(a_val, b_val, c_val, input_dtype, output_dtype, m, n, k,
              lhs_transpose, rhs_transpose, output_transpose, handle);
}

template <>
void CublasMatmul<__nv_bfloat16, __nv_bfloat16, float>::Run(
    const __nv_bfloat16 *a_val, const __nv_bfloat16 *b_val,
    __nv_bfloat16 *c_val) {
  auto input_dtype = ctype_to_cublas_dtype<__nv_bfloat16>::value;
  auto output_dtype = ctype_to_cublas_dtype<__nv_bfloat16>::value;
  run_SgemmEx(a_val, b_val, c_val, input_dtype, output_dtype, m, n, k,
              lhs_transpose, rhs_transpose, output_transpose, handle);
}

//===----------------------------------------------------------------------===//
// cublasHgemm
//===----------------------------------------------------------------------===//
template <>
void CublasMatmul<__half, __half, __half>::Run(const __half *a_val,
                                               const __half *b_val,
                                               __half *c_val) {
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

