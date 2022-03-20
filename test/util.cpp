#include "util.h"

template <typename T>
void test_rand_cpu_buffer(size_t size) {
  T* mat = (T*)malloc(size * sizeof(T));
  RandCPUBuffer(mat, size);
  PrintCPUBuffer(mat, size);
  free(mat);
}

template <typename T>
void test_rand_cpu_buffer(size_t size, float lb, float ub) {
  T* mat = (T*)malloc(size * sizeof(T));
  RandCPUBuffer(mat, size, lb, ub);
  PrintCPUBuffer(mat, size);
  free(mat);
}

template <typename T>
void test_fill_cpu_buffer(size_t size) {
  T* mat = (T*)malloc(size * sizeof(T));
  FillCPUBuffer(mat, size);
  PrintCPUBuffer(mat, size);
  free(mat);
}

int main() {
  test_rand_cpu_buffer<float>(10);
  test_rand_cpu_buffer<__half>(10);
  test_rand_cpu_buffer<__nv_bfloat16>(10);
  test_rand_cpu_buffer<float>(10, -1.f, 1.f);
  test_rand_cpu_buffer<__half>(10, -1.f, 1.f);
  test_rand_cpu_buffer<__nv_bfloat16>(10, -1.f, 1.f);

  test_fill_cpu_buffer<float>(10);
  test_fill_cpu_buffer<__half>(10);
  test_fill_cpu_buffer<__nv_bfloat16>(10);
  return 0;
}