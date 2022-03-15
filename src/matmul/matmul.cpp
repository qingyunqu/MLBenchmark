#include "matmul/check_matmul.h"
#include "matmul/cublas_matmul.h"
#include "util.h"
#include <stdio.h>

template <typename T, typename CompOn = float>
void benchmark(const T *a_val, const T *b_val, T *c_val, int64_t m, int64_t n,
               int64_t k, bool lhs_transpose, bool rhs_transpose,
               bool output_transpose) {
  cudaStream_t stream = nullptr;
  cudaEvent_t start, stop;
  cublasHandle_t handle;
  int run_time = 20;
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));

  CublasMatmul<T, CompOn> matmul;
  for (int i = 0; i < 10; i++) { // warm up
    matmul.Run(a_val, b_val, c_val, m, n, k, lhs_transpose, rhs_transpose,
               output_transpose, handle);
  }
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < run_time; i++) {
    matmul.Run(a_val, b_val, c_val, m, n, k, lhs_transpose, rhs_transpose,
               output_transpose, handle);
  }
  CUDACHECK(cudaEventRecord(stop, stream));

  CUDACHECK(cudaEventSynchronize(stop));
  float elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("%dx%dx%d, l:%d, r:%d, o:%d, time: %fms\n", m, n, k, lhs_transpose,
         rhs_transpose, output_transpose, elapsedTime / run_time);
  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  CUBLASCHECK(cublasDestroy(handle));
}

template <typename T, typename CompOn = float>
void test(const T *a_val, const T *b_val, T *c_val, int64_t m, int64_t n,
          int64_t k, bool lhs_transpose, bool rhs_transpose,
          bool output_transpose, float eps) {
  cudaStream_t stream = nullptr;
  cublasHandle_t handle;
  CUDACHECK(cudaSetDevice(0));
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  CublasMatmul<T, CompOn> matmul;
  matmul.Run(a_val, b_val, c_val, m, n, k, lhs_transpose, rhs_transpose,
             output_transpose, handle);
  CUDACHECK(cudaDeviceSynchronize());
  CheckMatmul(a_val, b_val, c_val, m, n, k, lhs_transpose, rhs_transpose,
              output_transpose, eps);
  CUBLASCHECK(cublasDestroy(handle));
}

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

  test<T, CompOn>(a, b, c, m, n, k, lhs_transpose, rhs_transpose,
                  output_transpose, eps);
  benchmark<T, CompOn>(a, b, c, m, n, k, lhs_transpose, rhs_transpose,
                       output_transpose);

  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
}

int main() {
  Run<float>(1024, 1024, 1024, false, false, false, 1e-3f);
  Run<float>(512, 512, 512, false, false, false, 1e-3f);
  Run<float>(511, 511, 511, false, false, false, 1e-3f);

  Run<__half>(1024, 1024, 1024, false, false, false, 2e-1f);
  Run<__half>(512, 512, 512, false, false, false, 2e-1f);
  Run<__half>(511, 511, 511, false, false, false, 2e-1f);

  Run<__half, __half>(1024, 1024, 1024, false, false, false, 5e-1f);
  Run<__half, __half>(512, 512, 512, false, false, false, 5e-1f);
  Run<__half, __half>(511, 511, 511, false, false, false, 5e-1f);

  Run<__nv_bfloat16>(1024, 1024, 1024, false, false, false, 5e-1f);
  Run<__nv_bfloat16>(512, 512, 512, false, false, false, 5e-1f);
  Run<__nv_bfloat16>(511, 511, 511, false, false, false, 5e-1f);
  return 0;
}