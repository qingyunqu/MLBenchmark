#include "matmul/cublas_matmul.h"
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
               output_transpose);
  }
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < run_time; i++) {
    matmul.Run(a_val, b_val, c_val, m, n, k, lhs_transpose, rhs_transpose,
               output_transpose);
  }
  CUDACHECK(cudaEventRecord(stop, stream));

  CUDACHECK(cudaEventSynchronize(stop));
  float elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("time: %fms\n", elapsedTime / run_time);
}

int main() {}