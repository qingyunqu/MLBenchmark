#pragma once

template <typename T> class Matmul {
public:
  Matmul() = default;
  virtual void Run(const T *a_val, const T *b_val, T *c_val, int64_t m,
                   int64_t n, int64_t k, bool lhs_transpose, bool rhs_transpose,
                   bool output_transpose) = 0;
  virtual ~Matmul() = default;
};
