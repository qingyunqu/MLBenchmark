#include "benchmark.h"
#include "convolution/CudnnConv.h"
#include "convolution/check_conv.h"
#include "util/kernel.h"
#include "util/util.h"
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
  RandCUDABuffer(c, N * oC * oH * oW);
  CUDACHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  cudnnHandle_t handle;
  CUDNNCHECK(cudnnCreate(&handle));
  CUDNNCHECK(cudnnSetStream(handle, stream));

  auto *op = new CudnnConv<T, To, CompOn>(layout, N, iC, iH, iW, oC, kH, kW, oH,
                                          oW, strideH, strideW, paddingH,
                                          paddingW, dilateH, dilateW, handle);
  op->AllocWorkspace();
  op->SetArgument(a, b, c);

  if (test) {
    // test
    op->Run();
    CUDACHECK(cudaDeviceSynchronize());
    bool passed = CheckConv<T, To, CompOn>(
        a, b, c, layout, N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW,
        paddingH, paddingW, dilateH, dilateW, eps);
    std::cout << "Test " << (passed ? "Passed" : "Failed") << ".\n";
  } else {
    // benchmark
    float time = benchmark(op, stream);
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

template <typename T, typename To = T, typename CompOn = float>
void RunBias(const std::string &layout, int64_t N, int64_t iH, int64_t iW,
             int64_t iC, int64_t oC, int64_t kH, int64_t kW, int64_t strideH,
             int64_t strideW, int64_t paddingH, int64_t paddingW) {
  int64_t dilateH = 1;
  int64_t dilateW = 1;
  int64_t oH = (iH + 2 * paddingH - dilateH * (kH - 1) - 1) / strideH + 1;
  int64_t oW = (iW + 2 * paddingW - dilateW * (kW - 1) - 1) / strideW + 1;
  T *a, *b;
  To *bias, *c, *ref_c;

  CUDACHECK(cudaMalloc(&a, N * iC * iH * iW * sizeof(T)));
  CUDACHECK(cudaMalloc(&b, oC * iC * kH * kW * sizeof(T)));
  CUDACHECK(cudaMalloc(&bias, oC * sizeof(To)));
  CUDACHECK(cudaMalloc(&c, N * oC * oH * oW * sizeof(To)));
  CUDACHECK(cudaMalloc(&ref_c, N * oC * oH * oW * sizeof(To)));
  RandCUDABuffer(a, N * iC * iH * iW, -1.f, 1.f);
  RandCUDABuffer(b, oC * iC * kH * kW, -1.f, 1.f);
  RandCUDABuffer(bias, oC, -1.f, 1.f);
  RandCUDABuffer(c, N * oC * oH * oW);
  FillCUDABuffer(ref_c, N * oC * oH * oW);
  CUDACHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  cudnnHandle_t handle;
  CUDNNCHECK(cudnnCreate(&handle));
  CUDNNCHECK(cudnnSetStream(handle, stream));

  auto *op = new CudnnConv<T, To, CompOn>(layout, N, iC, iH, iW, oC, kH, kW, oH,
                                          oW, strideH, strideW, paddingH,
                                          paddingW, dilateH, dilateW, handle);
  op->AllocWorkspace();
  op->SetArgument(a, b, ref_c);
  op->Run();
  BiasAdd(bias, ref_c, N * oH * oW, oC, stream);
  Sigmoid(ref_c, N * oH * oW * oC, stream);
  CUDACHECK(cudaDeviceSynchronize());

  auto *op1 = new CudnnConvBias<T, To, CompOn>(
      layout, N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH,
      paddingW, dilateH, dilateW, handle, EpilogueEnum::None);
  op1->AllocWorkspace();
  op1->SetArgument(a, b, bias, c);
  op1->Run();
  auto *act =
      new CudnnActivation<To>({N, oH, oW, oC}, EpilogueEnum::Sigmoid, handle);
  act->SetArgument(c);
  act->Run();

  delete op;
  delete op1;
  delete act;

  bool passed =
      CheckCUDABuffer<To>(c, ref_c, N * oH * oW * oC, 1e-3f, 1e-2f, 20);
  std::cout << "Test " << (passed ? "Passed" : "Failed") << ".\n";

  CUDNNCHECK(cudnnDestroy(handle));
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(bias));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
}

int main(int argc, char *argv[]) {
  // bool test = std::string(argv[1]) == "0" ? true : false;
  bool test = false;

  // for (int64_t b : {1, 8, 16, 32, 64, 128}) {
  // Run<__half>(/*layout=*/"NHWC", /*N*/ 128, /*iH*/ 56, /*iW*/ 56, /*iC*/ 64,
  //             /*oC*/ 64, /*kH*/ 3, /*kW*/ 3,
  //             /*strideH*/ 1, /*strideW*/ 1, /*paddingH*/ 1, /*paddingW*/ 1,
  //             1e-3f, test);
  //}

  // Run<__half>(/*layout=*/"NHWC", /*N*/ 128, /*iH*/ 224, /*iW*/ 224, /*iC*/ 3,
  //             /*oC*/ 64, /*kH*/ 7, /*kW*/ 7,
  //             /*strideH*/ 2, /*strideW*/ 2, /*paddingH*/ 3, /*paddingW*/ 3,
  //             1e-3f, test);

  RunBias<__half>(/*layout=*/"NHWC", /*N*/ 128, /*iH*/ 224, /*iW*/ 224,
                  /*iC*/ 3,
                  /*oC*/ 64, /*kH*/ 7, /*kW*/ 7,
                  /*strideH*/ 2, /*strideW*/ 2, /*paddingH*/ 3,
                  /*paddingW*/ 3);

  return 0;
}
