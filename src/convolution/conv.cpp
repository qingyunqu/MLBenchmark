#include "benchmark.h"
#include "convolution/check_conv.h"
#include "convolution/cudnn_conv.h"
#include "util.h"
#include <iostream>
#include <stdio.h>
#include <string>

template <typename T, typename To = T, typename CompOn = float>
void Run(const std::string &layout, int64_t N, int64_t iH, int64_t iW,
         int64_t iC, int64_t oC, int64_t kH, int64_t kW, int64_t strideH,
         int64_t strideW, int64_t paddingH, int64_t paddingW, float eps,
         bool test) {
  int64_t dilateH = 1;
  int64_t dilateW = 1;
  int64_t oH = (iH + 2 * paddingH - dilateH * (kH - 1) - 1) / strideH + 1;
  int64_t oW = (iW + 2 * paddingW - dilateW * (kW - 1) - 1) / strideW + 1;
  T *a, *b;
  To *c;

  CUDACHECK(cudaMalloc(&a, N * iC * iH * iW * sizeof(T)));
  CUDACHECK(cudaMalloc(&b, oC * iC * kH * kW * sizeof(T)));
  CUDACHECK(cudaMalloc(&c, N * oC * oH * oW * sizeof(To)));
  RandCUDABuffer(a, N * iC * iH * iW);
  RandCUDABuffer(b, oC * iC * kH * kW);
  CUDACHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  cudnnHandle_t handle;
  CUDNNCHECK(cudnnCreate(&handle));
  CUDNNCHECK(cudnnSetStream(handle, stream));

  Conv<T, To> *op = new CudnnConv<T, To, CompOn>(
      layout, N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH,
      paddingW, dilateH, dilateW, handle);

  if (test) {
    // test
    op->Run(a, b, c);
    CUDACHECK(cudaDeviceSynchronize());
    bool passed = CheckConv<T, To, CompOn>(
        a, b, c, layout, N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW,
        paddingH, paddingW, dilateH, dilateW, eps);
    if (passed) {
      printf("Test Passed.\n");
    }
  } else {
    // benchmark
    float time = benchmark<T, To>(op, stream, a, b, c);
    printf("layout: %s, N: %d, iH: %d, iW %d, iC: %d, oC: %d, kH: %d, kW: %d, "
           "oH: %d, oW: %d, "
           "strideH: %d, strideW: %d, paddingH: %d, paddingW: %d, dilateH: %d, "
           "dilateW: %d, time: %f ms\n",
           layout.c_str(), N, iH, iW, iC, oC, kH, kW, oH, oW, strideH, strideW,
           paddingH, paddingW, dilateH, dilateW, time);
  }

  delete op;
  CUDNNCHECK(cudnnDestroy(handle));
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
}

int main(int argc, char *argv[]) {
  // bool test = std::string(argv[1]) == "0" ? true : false;
  bool test = false;

  for (int64_t b : {1, 8, 16, 32, 64, 128}) {
    Run<__half>(/*layout=*/"NHWC", /*N*/ b, /*iH*/ 7, /*iW*/ 7, /*iC*/ 512,
                /*oC*/ 512, /*kH*/ 3, /*kW*/ 3,
                /*strideH*/ 1, /*strideW*/ 1, /*paddingH*/ 1, /*paddingW*/ 1,
                1e-3f, test);
  }

  return 0;
}
