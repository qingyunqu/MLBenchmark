#include "matmul/check_matmul.h"
#include "matmul/cublas_matmul.h"
//#include "matmul/cutlass_matmul.h"
#include "benchmark.h"
#include "util.h"
#include <stdio.h>

template <typename T, typename CompOn = float>
void Run(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
         bool rhs_transpose, bool output_transpose, float eps) {
  T *a, *b, *c;
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(T)));
  CUDACHECK(cudaMalloc(&b, k * n * sizeof(T)));
  CUDACHECK(cudaMalloc(&c, m * n * sizeof(T)));
  RandCUDABuffer(a, m * k);
  RandCUDABuffer(b, k * n);
  RandCUDABuffer(c, m * n);
  CUDACHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));

  Matmul<T> *op = new CublasMatmul<T, CompOn>(
      m, n, k, lhs_transpose, rhs_transpose, output_transpose, handle);
  // Matmul<T> *op = new CutlassMatmul<T, CompOn>(lhs_transpose, rhs_transpose,
  //                                              output_transpose, stream);

  // test
  op->Run(a, b, c);
  CUDACHECK(cudaDeviceSynchronize());
  CheckMatmul<T, CompOn>(a, b, c, m, n, k, lhs_transpose, rhs_transpose,
                         output_transpose, eps);
  // benchmark
  float time = benchmark<T>(op, stream, a, b, c);
  printf("%dx%dx%d, l:%d, r:%d, o:%d, time: %fms\n", m, n, k, lhs_transpose,
         rhs_transpose, output_transpose, time);

  delete op;
  CUBLASCHECK(cublasDestroy(handle));
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
}

int main() {
  Run<float>(1024, 1024, 1024, false, false, false, 1e-3f);
  Run<float>(512, 512, 512, false, false, false, 1e-3f);
  Run<float>(511, 511, 511, false, false, false, 1e-3f);

  Run<__half>(1024, 1024, 1024, false, false, false, 5e-1f);
  Run<__half>(512, 512, 512, false, false, false, 5e-1f);
  Run<__half>(511, 511, 511, false, false, false, 5e-1f);

  // Run<__half, __half>(1024, 1024, 1024, false, false, false, 5e-1f);
  // Run<__half, __half>(512, 512, 512, false, false, false, 5e-1f);
  // Run<__half, __half>(511, 511, 511, false, false, false, 5e-1f);

  // Run<__nv_bfloat16>(1024, 1024, 1024, false, false, false, 5e-1f);
  // Run<__nv_bfloat16>(512, 512, 512, false, false, false, 5e-1f);
  // Run<__nv_bfloat16>(511, 511, 511, false, false, false, 5e-1f);
  return 0;
}