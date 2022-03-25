#pragma once

template <typename T>
__global__ void bias_add(T *bias, T *result, int64_t m, int64_t n);

template <typename T> __global__ void relu(T *result, int64_t size);

template <typename T> __global__ void sigmoid(T *result, int64_t size);