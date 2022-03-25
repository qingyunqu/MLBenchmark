#include "manifest.h"

#include <cuda_fp16.h>

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main() {
  Manifest manifest;
  cutlass::library::initialize_all_gemm_operations(manifest);

  // manifest.template profile_gemm<__half, __half, float, float>(
  //     2045, 2048, 2048, LayoutEnum::ColumnMajor, LayoutEnum::ColumnMajor,
  //     LayoutEnum::RowMajor);
  // manifest.template profile_gemm<__half, __half, float, float>(
  //     2048, 2048, 2048, LayoutEnum::RowMajor, LayoutEnum::RowMajor,
  //     LayoutEnum::RowMajor);

  // manifest.template profile_gemm<__half, __half, __half, float>(
  //     2048, 2048, 2048, LayoutEnum::ColumnMajor, LayoutEnum::ColumnMajor,
  //     LayoutEnum::RowMajor);
  manifest.template profile_gemm<__half, __half, __half, float>(
      4096, 4096, 4096, LayoutEnum::RowMajor, LayoutEnum::RowMajor,
      LayoutEnum::RowMajor);
  return 0;
}