#include "Manifest.h"
#include "profile.h"

#include <cuda_fp16.h>

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main() {
  Manifest manifest;
  cutlass::library::initialize_all_gemm_operations(manifest);

  profile_gemm_bias<__half, __half, float, float>(
      manifest, 2045, 2048, 2048, LayoutEnum::ColumnMajor,
      LayoutEnum::ColumnMajor, LayoutEnum::RowMajor);
  profile_gemm_bias<__half, __half, float, float>(
      manifest, 2048, 2048, 2048, LayoutEnum::RowMajor, LayoutEnum::RowMajor,
      LayoutEnum::RowMajor);

  profile_gemm_bias<__half, __half, __half, float>(
      manifest, 2048, 2048, 2048, LayoutEnum::ColumnMajor,
      LayoutEnum::ColumnMajor, LayoutEnum::RowMajor);
  profile_gemm_bias<__half, __half, __half, float>(
      manifest, 4096, 4096, 4096, LayoutEnum::RowMajor, LayoutEnum::RowMajor,
      LayoutEnum::RowMajor);
  return 0;
}