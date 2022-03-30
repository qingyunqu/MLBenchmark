#pragma once

#include <cuda_runtime.h>

template <typename T>
void BiasAdd(T *bias, T *result, int64_t m, int64_t n, cudaStream_t stream);

template <typename T> void Relu(T *result, int64_t size, cudaStream_t stream);

template <typename T>
void Sigmoid(T *result, int64_t size, cudaStream_t stream);

template <typename T> void Tanh(T *result, int64_t size, cudaStream_t stream);
