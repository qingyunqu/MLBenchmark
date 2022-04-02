#pragma once

#include "dtype.h"
#include "ops.h"
#include <cassert>
#include <cuda_runtime.h>

template <typename T>
void BiasAdd(T *bias, T *result, int64_t m, int64_t n, cudaStream_t stream);

template <typename T> void Relu(T *result, int64_t size, cudaStream_t stream);

template <typename T>
void Sigmoid(T *result, int64_t size, cudaStream_t stream);

template <typename T> void Tanh(T *result, int64_t size, cudaStream_t stream);

template <typename T> void Gelu(T *result, int64_t size, cudaStream_t stream);

template <typename T> class MyActivation : public Op<T> {
public:
  MyActivation(size_t size, EpilogueEnum epilogue, cudaStream_t stream)
      : size(size), epilogue(epilogue), stream(stream) {}

  virtual bool Check() { return true; }

  virtual void SetArgument(T *_input) { input = _input; }

  virtual void Run() {
    if (epilogue == EpilogueEnum::None) {
      return;
    } else if (epilogue == EpilogueEnum::Relu) {
      Relu(input, size, stream);
    } else if (epilogue == EpilogueEnum::Sigmoid) {
      Sigmoid(input, size, stream);
    } else if (epilogue == EpilogueEnum::Tanh) {
      Tanh(input, size, stream);
    } else if (epilogue == EpilogueEnum::Grelu) {
      Gelu(input, size, stream);
    } else {
      assert(false);
    }
  }

private:
  size_t size;
  EpilogueEnum epilogue;
  cudaStream_t stream;
  T *input;
};
