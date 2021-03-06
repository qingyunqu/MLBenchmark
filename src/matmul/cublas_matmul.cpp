#include "benchmark.h"
#include "matmul/CublasMatmul.h"
#include "matmul/check_matmul.h"
#include "util/util.h"
#include <iostream>
#include <stdio.h>
#include <string>

template <typename T, typename To = T, typename CompOn = float>
void Run(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
         bool rhs_transpose, bool output_transpose, float eps, bool test) {
  T *a, *b;
  To *c;
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(T)));
  CUDACHECK(cudaMalloc(&b, k * n * sizeof(T)));
  CUDACHECK(cudaMalloc(&c, m * n * sizeof(To)));
  RandCUDABuffer(a, m * k);
  RandCUDABuffer(b, k * n);

  cudaStream_t stream = nullptr;
  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));

  auto *op = new CublasMatmul<T, To, CompOn>(
      m, n, k, lhs_transpose, rhs_transpose, output_transpose, handle);
  op->SetArgument(a, b, c);

  if (test) {
    // test
    op->Run();
    CUDACHECK(cudaDeviceSynchronize());
    bool passed = CheckMatmul<T, To, CompOn>(
        a, b, c, m, n, k, lhs_transpose, rhs_transpose, output_transpose, eps);
    if (passed) {
      printf("Test Passed.\n");
    }
  } else {
    // benchmark
    float time = benchmark(op, stream);
    printf("%dx%dx%d, l:%d, r:%d, o:%d, time: %f ms\n", m, n, k, lhs_transpose,
           rhs_transpose, output_transpose, time);
  }

  delete op;
  CUBLASCHECK(cublasDestroy(handle));
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
}

int main(int argc, char *argv[]) {
  bool test = false;
  // bool test = std::string(argv[1]) == "0" ? true : false;

  Run<__half, __half, float>(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), false,
                             false, true, 1e-2f, /*test=*/false);

  // Ti To CompOn
  // for (bool lhs: {false, true}) {
  //   for (bool rhs: {false, true}) {
  //     for (int64_t i: {512, 1024, 2048, 3072, 4096, 5120}) {
  //       Run<__half, __half, float>(i, i, i, lhs, rhs, true, 1e-2f, test);
  //     }
  //   }
  // }

  // Run<__half, float, float>(1024, 1024, 1024, false, false, false, 1e-2f,
  // test);
  // Run<__half, float, float>(512, 512, 512, false, false, false, 1e-2f, test);
  // Run<__half, float, float>(511, 511, 511, false, false, false, 1e-2f, test);

  // Run<float>(1024, 1024, 1024, false, false, false, 1e-3f, test);
  // Run<float>(512, 512, 512, false, false, false, 1e-3f, test);
  // Run<float>(511, 511, 511, false, false, false, 1e-3f, test);

  // Run<__half>(1024, 1024, 1024, false, false, false, 5e-1f, test);
  // Run<__half>(512, 512, 512, false, false, false, 5e-1f, test);
  // Run<__half>(511, 511, 511, false, false, false, 5e-1f, test);

  // Run<__half, __half, __half>(1024, 1024, 1024, false, false, false, 5e-1f,
  // test);
  // Run<__half, __half, __half>(512, 512, 512, false, false, false, 5e-1f,
  // test);
  // Run<__half, __half, __half>(511, 511, 511, false, false, false, 5e-1f,
  // test);

  // Run<__nv_bfloat16>(1024, 1024, 1024, false, false, false, 5e-1f, test);
  // Run<__nv_bfloat16>(512, 512, 512, false, false, false, 5e-1f, test);
  // Run<__nv_bfloat16>(511, 511, 511, false, false, false, 5e-1f, test);
  return 0;
}