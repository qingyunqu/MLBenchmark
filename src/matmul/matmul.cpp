#include "matmul/matmul.h"
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
  printf("benchmark time: %fms\n", elapsedTime / run_time);
  CUDACHECK(cudaDeviceSynchronize());
}

template <typename T, typename CompOn = float>
void test(const T *a_val, const T *b_val, T *c_val, int64_t m, int64_t n,
          int64_t k, bool lhs_transpose, bool rhs_transpose,
          bool output_transpose) {
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
              output_transpose, 1e-4f);
}

template <typename T>
void Run(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
         bool rhs_transpose, bool output_transpose) {
  T *a, *b, *c;
  CUDACHECK(cudaMalloc(&a, m * k * sizeof(T)));
  CUDACHECK(cudaMalloc(&b, k * n * sizeof(T)));
  CUDACHECK(cudaMalloc(&c, m * n * sizeof(T)));
  RandCUDABuffer(a, m * k);
  RandCUDABuffer(b, k * n);
  RandCUDABuffer(c, m * n);

  test<T>(a, b, c, m, n, k, lhs_transpose, rhs_transpose, output_transpose);
  benchmark<T>(a, b, c, m, n, k, lhs_transpose, rhs_transpose,
               output_transpose);
}

int main() {
  Run<float>(1024, 1024, 1024, false, false, false);
  return 0;
}