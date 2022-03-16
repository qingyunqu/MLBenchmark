#pragma once

template <typename T> class Matmul {
public:
  Matmul(bool lhs_transpose, bool rhs_transpose, bool output_transpose)
      : lhs_transpose(lhs_transpose), rhs_transpose(rhs_transpose),
        output_transpose(output_transpose) {}
  virtual void Run(const T *a_val, const T *b_val, T *c_val, int64_t m,
                   int64_t n, int64_t k) = 0;
  virtual ~Matmul() = default;

public:
  bool lhs_transpose, rhs_transpose, output_transpose;
};
