#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../check.h"
#include "util/util.h"

template <typename T, typename To, typename CompOn = float>
bool CheckMatmul(const T *d_A, const T *d_B, To *d_C, int64_t m, int64_t n,
                 int64_t k, bool lhs_transpose, bool rhs_transpose,
                 bool output_transpose, float eps) {
  T *h_A = (T *)malloc(m * k * sizeof(T));
  T *h_B = (T *)malloc(k * n * sizeof(T));
  To *h_C = (To *)malloc(m * n * sizeof(To));
  CUDACHECK(cudaMemcpy(h_A, d_A, m * k * sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_B, d_B, k * n * sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(To), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

  bool check = true;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      CompOn sum = static_cast<CompOn>(0.0f);
      for (int64_t l = 0; l < k; ++l) {
        if (!lhs_transpose && !rhs_transpose) {
          auto temp = static_cast<CompOn>(h_A[i * k + l]) *
                      static_cast<CompOn>(h_B[l * n + j]);
          sum = sum + static_cast<CompOn>(temp);
        } else if (lhs_transpose && !rhs_transpose) {
          auto temp = static_cast<CompOn>(h_A[l * m + i]) *
                      static_cast<CompOn>(h_B[l * n + j]);
          sum = sum + static_cast<CompOn>(temp);
        } else if (!lhs_transpose && rhs_transpose) {
          auto temp = static_cast<CompOn>(h_A[i * k + l]) *
                      static_cast<CompOn>(h_B[j * k + l]);
          sum = sum + static_cast<CompOn>(temp);
        } else {
          auto temp = static_cast<CompOn>(h_A[l * m + i]) *
                      static_cast<CompOn>(h_B[j * k + l]);
          sum = sum + static_cast<CompOn>(temp);
        }
      }
      if (!output_transpose) {
        check = EXPECT_NEAR(h_C[i * n + j], static_cast<To>(sum), 1e-3f, eps);
      } else {
        check = EXPECT_NEAR(h_C[i + j * m], static_cast<To>(sum), 1e-3f, eps);
      }
      if (!check) {
        goto EXIT;
      }
    }
  }

EXIT:
  free(h_A);
  free(h_B);
  free(h_C);
  return check;
}
