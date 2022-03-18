#pragma once

#include <cuda_runtime.h>

#include "check.h"
#include "ops.h"

template <typename T, typename To, typename... Args>
float benchmark(Op<T, To> *op, cudaStream_t stream, Args... args) {
  cudaEvent_t start, stop;
  int run_time = 20;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  for (int i = 0; i < 10; i++) { // warm up
    op->Run(args...);
  }
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < run_time; i++) {
    op->Run(args...);
  }
  CUDACHECK(cudaEventRecord(stop, stream));

  CUDACHECK(cudaEventSynchronize(stop));
  float elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  return elapsedTime / run_time;
}
