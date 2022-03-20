#include "benchmark.h"
#include "matmul/check_matmul.h"
#include "matmul/cublas_matmul.h"
// #include "matmul/cutlass_matmul.h"
#include "util.h"
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
  CUDACHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));

  Matmul<T, To> *op = new CublasMatmul<T, To, CompOn>(
      m, n, k, lhs_transpose, rhs_transpose, output_transpose, handle);
  // Matmul<T, To> *op = new CutlassMatmul<T, To, CompOn>(
  //     m, n, k, lhs_transpose, rhs_transpose, output_transpose, stream);

  if (test) {
    // test
    op->Run(a, b, c);
    CUDACHECK(cudaDeviceSynchronize());
    bool passed = CheckMatmul<T, To, CompOn>(
        a, b, c, m, n, k, lhs_transpose, rhs_transpose, output_transpose, eps);
    if (passed) {
      printf("Test Passed.\n");
    }
  } else {
    // benchmark
    float time = benchmark<T, To>(op, stream, a, b, c);
    printf("%dx%dx%d, l:%d, r:%d, o:%d, time: %fms\n", m, n, k, lhs_transpose,
           rhs_transpose, output_transpose, time);
  }

  delete op;
  CUBLASCHECK(cublasDestroy(handle));
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
}

int main(int argc, char *argv[]) {
  bool test = std::string(argv[1]) == "0" ? true : false;

  Run<__half, __half, float>(512, 512, 512, false, false, true, 1e-2f, test);
  Run<__half, __half, float>(1024, 1024, 1024, false, false, true, 1e-2f, test);
  Run<__half, __half, float>(2048, 2048, 2048, false, false, true, 1e-2f, test);
  Run<__half, __half, float>(3072, 3072, 3072, false, false, true, 1e-2f, test);
  // Run<__half, float, float>(4096, 4096, 4096, true, true, false, 1e-2f, test);

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