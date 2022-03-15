#pragma once

#include "./check.h"
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// generate random number between [0, 1) for floating point type
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void RandCPUBuffer(T *mat, size_t size, T lb = static_cast<T>(0.f),
                   T ub = static_cast<T>(1.f)) {
  for (size_t i = 0; i < size; ++i) {
    T temp = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    mat[i] = temp * (ub - lb) + lb;
  }
}

// generate random number between [0, ub) for integer type
template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
void RandCPUBuffer(T *mat, size_t size, size_t ub = RAND_MAX) {
  for (size_t i = 0; i < size; ++i) {
    mat[i] = static_cast<T>(rand()) % static_cast<T>(ub);
  }
}

template <typename T, typename... Args>
void RandCUDABuffer(T *mat, size_t size, Args... args) {
  T *h_ptr = (T *)malloc(size * sizeof(T));
  RandCPUBuffer<T>(h_ptr, size, args...);
  cudaMemcpy(mat, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
  CUDACHECK(cudaDeviceSynchronize());
  free(h_ptr);
}

template <typename... Args>
inline void RandCUDABuffer(__half *mat, size_t size, Args... args) {
  float *h_ptr_f = (float *)malloc(size * sizeof(float));
  RandCPUBuffer<float>(h_ptr_f, size, args...);
  __half *h_ptr = (__half *)malloc(size * sizeof(__half));
  for (size_t i = 0; i < size; i++) {
    h_ptr[i] = static_cast<__half>(h_ptr_f[i]);
  }
  cudaMemcpy(mat, h_ptr, size * sizeof(__half), cudaMemcpyHostToDevice);
  CUDACHECK(cudaDeviceSynchronize());
  free(h_ptr);
  free(h_ptr_f);
}

template <typename T> inline bool EXPECT_NEAR(T first, T second, float eps) {
  float diff = static_cast<float>(first) - static_cast<float>(second);
  if (std::abs(diff) > eps) {
    printf("diff error, first: %f, second: %f\n", static_cast<float>(first),
           static_cast<float>(second));
    return false;
  }
  return true;
}
