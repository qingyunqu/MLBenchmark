#pragma once

#include "./check.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace std {
template <> struct is_floating_point<__half> {
  static constexpr bool value = true;
};
template <> struct is_floating_point<__nv_bfloat16> {
  static constexpr bool value = true;
};
}

//===----------------------------------------------------------------------===//
// RandCPUBuffer
//===----------------------------------------------------------------------===//

// generate random number between [0, 1) for floating point type
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void RandCPUBuffer(T *mat, size_t size, float lb = 0.f, float ub = 1.f) {
  for (size_t i = 0; i < size; ++i) {
    float temp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    mat[i] = static_cast<T>(temp * (ub - lb) + lb);
  }
}

// generate random number between [0, ub) for integer type
template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
void RandCPUBuffer(T *mat, size_t size, size_t ub = RAND_MAX) {
  for (size_t i = 0; i < size; ++i) {
    mat[i] = static_cast<T>(rand()) % static_cast<T>(ub);
  }
}

//===----------------------------------------------------------------------===//
// RandCUDABuffer
//===----------------------------------------------------------------------===//

template <typename T, typename... Args>
void RandCUDABuffer(T *mat, size_t size, Args... args) {
  T *h_ptr = (T *)malloc(size * sizeof(T));
  RandCPUBuffer<T>(h_ptr, size, args...);
  CUDACHECK(cudaMemcpy(mat, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK(cudaDeviceSynchronize());
  free(h_ptr);
}

//===----------------------------------------------------------------------===//
// FillBuffer
//===----------------------------------------------------------------------===//

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void FillCPUBuffer(T* mat, size_t size, float value = 0.f) {
  for (size_t i = 0; i < size; i++) {
    mat[i] = static_cast<T>(value);
  }
}

template <typename T>
void FillCUDABuffer(T* mat, size_t size, float value = 0.f) {
  T* h_mat = (T*)malloc(size * sizeof(T));
  FillCPUBuffer<T>(h_mat, size, value);
  CUDACHECK(cudaMemcpy(mat, h_mat, size * sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK(cudaDeviceSynchronize());
  free(h_mat);
}

//===----------------------------------------------------------------------===//
// PrintBuffer
//===----------------------------------------------------------------------===//

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void PrintCPUBuffer(T* mat, size_t size, size_t print_size = 0) {
  print_size = (print_size == 0) ? size : print_size;
  print_size = (print_size > size) ? size : print_size;
  for (size_t i = 0; i < print_size; i++) {
    printf("%f ", static_cast<float>(mat[i]));
    if (i != 0 && i % 20 == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

template <typename T>
void PrintCUDABuffer(T* mat, size_t size, size_t print_size = 0) {
  T* h_mat = (T*)malloc(size * sizeof(T));
  CUDACHECK(cudaMemcpy(h_mat, mat, size * sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());
  PrintCPUBuffer<T>(h_mat, size, print_size);
  free(h_mat);
}

//===----------------------------------------------------------------------===//
// CheckBuffer
//===----------------------------------------------------------------------===//

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
bool EXPECT_NEAR(T first, T second, double absolute_eps, double relative_eps) {
  double first_val = static_cast<double>(first);
  double second_val = static_cast<double>(second);
  double abs_diff = std::abs(first_val - second_val);
  double rel_diff = 0;
  if (first_val != 0) {
    rel_diff = std::max(rel_diff, abs_diff / std::abs(first_val));
  }
  if (second_val != 0) {
    rel_diff = std::max(rel_diff, abs_diff / std::abs(second_val));
  }

  if (abs_diff > absolute_eps && rel_diff > relative_eps) {
    std::cerr << "ExpectNear Error: first value " << first_val
              << ", second value " << second_val << "\n";
    std::cerr << "Expected absolute eps: " << absolute_eps
              << ", real absolute: " << abs_diff
              << "; expected relative eps: " << relative_eps
              << ", real relative: " << rel_diff << "\n";
    return false;
  }
  return true;
}

template <typename T>
inline bool CheckCPUBuffer(T *first, T *second, size_t size, float relative_eps,
                           float absolute_eps, size_t print_count = 1) {
  size_t count = 0;
  for (size_t i = 0; i < size && count < print_count; i++) {
    if (!EXPECT_NEAR<T>(first[i], second[i], absolute_eps, relative_eps)) {
      count++;
    }
  }
  return count == 0;
}

template <typename T>
inline bool CheckCUDABuffer(T *first, T *second, size_t size,
                            float relative_eps, float absolute_eps,
                            size_t print_size = 1) {
  T* h_first = (T*)malloc(size * sizeof(T));
  T* h_second = (T*)malloc(size * sizeof(T));
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(h_first, first, size * sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_second, second, size * sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());
  bool passed = CheckCPUBuffer<T>(h_first, h_second, size, relative_eps,
                                  absolute_eps, print_size);
  free(h_first);
  free(h_second);
  return passed;
}