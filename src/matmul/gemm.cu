#include "matmul/manifest.h"

int main() {
  matmul::Manifest manifest;
  manifest.template append<cutlass::half_t, cutlass::layout::ColumnMajor,
                           cutlass::half_t, cutlass::layout::ColumnMajor, float,
                           cutlass::layout::RowMajor, float>(
      new MatmulOperation<
          Operation_cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8>(
          "cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8"));

  manifest.template profile<cutlass::half_t, cutlass::half_t, float, float>(
      1024, 1024, 1024, true, true, false);
  return 0;
}