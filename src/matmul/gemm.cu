#include "manifest.h"

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main() {
  Manifest manifest;
  cutlass::library::initialize_all_gemm_operations(manifest);

  manifest.template profile<cutlass::half_t, cutlass::half_t, float, float>(
      1024, 1024, 1024, LayoutEnum::ColumnMajor, LayoutEnum::ColumnMajor,
      LayoutEnum::RowMajor);
  return 0;
}