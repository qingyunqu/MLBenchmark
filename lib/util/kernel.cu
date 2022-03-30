#include "check.h"
#include "util/kernel.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
__global__ void bias_add(T *bias, T *result, int64_t m, int64_t n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n * m) {
    result[id] += bias[id % n];
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
#if (__CUDACC_VER_MAJOR__ >= 10) && (__CUDA_ARCH__ >= 750)
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

template <typename T> __global__ void tanh(T *result, int64_t size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    result[id] = static_cast<T>(::tanh(static_cast<float>(result[id])));
  }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
void BiasAdd(T *bias, T *result, int64_t m, int64_t n, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((m * n + block.x - 1) / block.x);
  bias_add<T><<<grid, block, 0, stream>>>(bias, result, m, n);
  after_kernel_launch();
}

template <typename T> void Relu(T *result, int64_t size, cudaStream_t stream) {
  relu<T><<<(size + 256 - 1) / 256, 256, 0, stream>>>(result, size);
  after_kernel_launch();
}

template <typename T>
void Sigmoid(T *result, int64_t size, cudaStream_t stream) {
  sigmoid<T><<<(size + 256 - 1) / 256, 256, 0, stream>>>(result, size);
  after_kernel_launch();
}

template <typename T> void Tanh(T *result, int64_t size, cudaStream_t stream) {
  tanh<T><<<(size + 256 - 1) / 256, 256, 0, stream>>>(result, size);
  after_kernel_launch();
}

template void BiasAdd<float>(float *, float *, int64_t, int64_t, cudaStream_t);
template void BiasAdd<__half>(__half *, __half *, int64_t, int64_t,
                              cudaStream_t);
template void Relu<float>(float *, int64_t, cudaStream_t);
template void Relu<__half>(__half *, int64_t, cudaStream_t);
template void Sigmoid<float>(float *, int64_t, cudaStream_t);
template void Sigmoid<__half>(__half *, int64_t, cudaStream_t);
template void Tanh<float>(float *, int64_t, cudaStream_t);
template void Tanh<__half>(__half *, int64_t, cudaStream_t);