#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "check.h"
#include "ops.h"

constexpr int warm_up = 5;
constexpr float run_time = 600; // ms

template <typename Op, typename... Args>
float benchmark(Op *op, cudaStream_t stream, Args... args) {
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  float elapsedTime = 0.f;

  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < warm_up; i++) { // warm up
    op->Run(args...);
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaEventSynchronize(stop));
  elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  int run = std::ceil(run_time / (elapsedTime / warm_up));
  // printf("run: %d\n", run);
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < run; i++) {
    op->Run(args...);
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaEventSynchronize(stop));
  elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  return elapsedTime / run;
}

template <typename Gemm>
float benchmark_cutlass(Gemm* op, cudaStream_t stream) {
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  float elapsedTime = 0.f;

  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < warm_up; i++) { // warm up
    (*op)();
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaEventSynchronize(stop));
  elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  int run = std::ceil(run_time / (elapsedTime / warm_up));
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < run; i++) {
    (*op)();
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaEventSynchronize(stop));
  elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  return elapsedTime / run;
}
