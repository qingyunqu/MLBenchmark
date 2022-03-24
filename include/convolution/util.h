#pragma once

#include "../util.h"
#include "check.h"

template <typename TA, typename TB, typename TC>
void InitConv2dTensor(int64_t N, int64_t iH, int64_t iW, int64_t iC, int64_t oH,
                      int64_t oW, int64_t oC, int64_t kH, int64_t kW, TA *&a,
                      TB *&b, TC *&c, TC *&ref_c) {
  CUDACHECK(cudaMalloc(&a, N * iH * iW * iC * sizeof(TA)));
  CUDACHECK(cudaMalloc(&b, oC * kH * kW * iC * sizeof(TB)));
  CUDACHECK(cudaMalloc(&c, N * oH * oW * oC * sizeof(TC)));
  CUDACHECK(cudaMalloc(&ref_c, N * oH * oW * oC * sizeof(TC)));
  RandCUDABuffer(a, N * iH * iW * iC, -1.f, 1.f);
  RandCUDABuffer(b, oC * kH * kW * iC, -1.f, 1.f);
  RandCUDABuffer(c, N * oH * oW * oC);
  FillCUDABuffer(ref_c, N * oH * oW * oC);
}
