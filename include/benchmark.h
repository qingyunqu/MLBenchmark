#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <unistd.h>

#include "check.h"
#include "ops.h"

constexpr int warm_up = 5;
constexpr int run = 100;
constexpr float run_time = 600; // ms

template <typename Op, typename... Args>
float benchmark(Op *op, cudaStream_t stream, Args... args) {
  usleep(50 * 1000);

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

  // int run = std::ceil(run_time / (elapsedTime / warm_up));
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

template <typename Op0, typename Op1, typename... Args>
float benchmark2(Op0 *op0, Op1 *op1, cudaStream_t stream, Args... args) {
  usleep(50 * 1000);

  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  float elapsedTime = 0.f;

  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < warm_up; i++) { // warm up
    op0->Run(args...);
    op1->Run();
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaEventSynchronize(stop));
  elapsedTime = 0.f;
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  // int run = std::ceil(run_time / (elapsedTime / warm_up));
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < run; i++) {
    op0->Run(args...);
    op1->Run();
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
  usleep(50 * 1000);

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

  // int run = std::ceil(run_time / (elapsedTime / warm_up));
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
