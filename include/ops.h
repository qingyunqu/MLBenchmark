#pragma once

template <typename T> class Matmul {
public:
  Matmul(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
         bool rhs_transpose, bool output_transpose)
      : m(m), n(n), k(k), lhs_transpose(lhs_transpose),
        rhs_transpose(rhs_transpose), output_transpose(output_transpose) {}
  virtual void Run(const T *a_val, const T *b_val, T *c_val) = 0;
  virtual ~Matmul() = default;

public:
  int64_t m, n, k;
  bool lhs_transpose, rhs_transpose, output_transpose;
};

template <typename T> class Conv {
public:
  Conv(const string &layout, int64_t N, int64_t iC, int64_t iH, int64_t iW,
       int64_t oC, int64_t kH, int64_t kW, int64_t oH, int64_t oW,
       int64_t strideH, int64_t strideW, int64_t paddingH, int64_t paddingW,
       int64_t dilateH, int64_t dilateW)
      : layout(layout), N(N), iC(iC), iH(iH), iW(iW), oC(oC), kH(kH), kW(kW),
        oH(oH), oW(oW), strideH(strideH), strideW(strideW), paddingH(paddingH),
        paddingW(paddingW), dilateH(dilateH), dilateW(dilateW) {}
  virtual bool Check() = 0;
  virtual void Run(const T *input, const T *filter, T *output) = 0;
  virtual ~Conv() = default;

public:
  string layout;
  int64_t N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH,
      paddingW, dilateH, dilateW;
};