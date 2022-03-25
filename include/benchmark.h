#pragma once

#include <cuda_runtime.h>

#include "check.h"
#include "ops.h"

constexpr int warm_up = 50;
constexpr int run_time = 50;

template <typename Op, typename... Args>
float benchmark(Op *op, cudaStream_t stream, Args... args) {
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warm_up; i++) { // warm up
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

template <typename Gemm>
float benchmark_cutlass(Gemm* op, cudaStream_t stream) {
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warm_up; i++) { // warm up
    (*op)();
  }
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < run_time; i++) {
    (*op)();
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
