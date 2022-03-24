#include "manifest.h"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main() {
  Manifest manifest;
  cutlass::library::initialize_all_gemm_operations(manifest);

  manifest.template profile<__half, __half, float, float>(
      2048, 2048, 2048, LayoutEnum::ColumnMajor, LayoutEnum::ColumnMajor,
      LayoutEnum::RowMajor);
  manifest.template profile<__half, __half, float, float>(
      2048, 2048, 2048, LayoutEnum::RowMajor, LayoutEnum::RowMajor,
      LayoutEnum::RowMajor);

  manifest.template profile<__half, __half, __half, float>(
      2048, 2048, 2048, LayoutEnum::ColumnMajor, LayoutEnum::ColumnMajor,
      LayoutEnum::RowMajor);
  manifest.template profile<__half, __half, __half, float>(
      2048, 2048, 2048, LayoutEnum::RowMajor, LayoutEnum::RowMajor,
      LayoutEnum::RowMajor);
  return 0;
}