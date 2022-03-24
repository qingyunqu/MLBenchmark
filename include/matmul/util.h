#pragma once

#include "../util.h"
#include "check.h"

template <typename TA, typename TB, typename TC>
void InitMatmulTensor(int64_t m, int64_t n, int64_t k, TA *&a, TB *&b, TC *&c,
                      TC *&ref_c) {
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(TA)));
  CUDACHECK(cudaMalloc(&b, n * k * sizeof(TB)));
  CUDACHECK(cudaMalloc(&c, m * n * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_c, m * n * sizeof(TC)));
  RandCUDABuffer(a, m * k);
  RandCUDABuffer(b, n * k);
  RandCUDABuffer(c, m * n);
  FillCUDABuffer(ref_c, m * n);
}
