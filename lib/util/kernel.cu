#include "util/kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void bias_add(T *bias, T *result, int64_t m, int64_t n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int column = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && column < m) {
    result[column * n + row] += bias[row];
  }
}

template <typename T> __global__ void relu(T *result, int64_t size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    result[id] =
        result[id] >= static_cast<T>(0) ? result[id] : static_cast<T>(0);
  }
}

template <typename T> __global__ void sigmoid(T *result, int64_t size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    result[id] =
        static_cast<T>(1.f / (1.f + exp(-static_cast<double>(result[id]))));
  }
}

template __global__ void bias_add<float>(float *, float *, int64_t, int64_t);
template __global__ void bias_add<__half>(__half *, __half *, int64_t, int64_t);
template __global__ void relu<float>(float *, int64_t);
template __global__ void relu<__half>(__half *, int64_t);
template __global__ void sigmoid<float>(float *, int64_t);
template __global__ void sigmoid<__half>(__half *, int64_t);