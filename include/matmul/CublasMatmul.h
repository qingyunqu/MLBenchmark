#pragma once

#include <cublas_v2.h>

#include "ops.h"

const char *cublasGetErrorString(cublasStatus_t status);

template <typename T, typename To, typename CompOn>
class CublasMatmul : public Op<T, To> {
public:
  CublasMatmul(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
               bool rhs_transpose, bool output_transpose,
               cublasHandle_t handle);
  virtual bool Check() override { return true; }
  virtual void SetArgument(T *a, T *b, To *c) override {
    a_val = a;
    b_val = b;
    c_val = c;
  }
  virtual void Run() override;
  virtual ~CublasMatmul() = default;

private:
  cublasHandle_t handle;
  int64_t m, n, k;
  bool lhs_transpose, rhs_transpose, output_transpose;
  T *a_val = nullptr, *b_val = nullptr;
  To *c_val = nullptr;
};
