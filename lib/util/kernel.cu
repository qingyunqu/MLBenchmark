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

__device__ inline float fast_exp(float x) { return ::exp(x); }

__device__ inline float fast_exp(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 10) &&                  \
    (__CUDA_ARCH__ >= 750)
  return ::hexp(x);
#else
  return fast_exp(float(x));
#endif
}

template <typename T> __global__ void sigmoid(T *result, int64_t size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    result[id] = T(1) / (T(1) + static_cast<T>(fast_exp(-result[id])));
  }
}

template __global__ void bias_add<float>(float *, float *, int64_t, int64_t);
template __global__ void bias_add<__half>(__half *, __half *, int64_t, int64_t);
template __global__ void relu<float>(float *, int64_t);
template __global__ void relu<__half>(__half *, int64_t);
template __global__ void sigmoid<float>(float *, int64_t);
template __global__ void sigmoid<__half>(__half *, int64_t);