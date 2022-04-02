#include "Manifest.h"
#include "profile.h"

#include <cassert>
#include <cuda_fp16.h>
#include <string>

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main(int argc, char *argv[]) {
  assert(argc == 9 || argc == 10);
  std::string output_type(argv[1]);
  std::string accmu_type(argv[2]);
  std::string lhs_layout(argv[3]);
  std::string rhs_layout(argv[4]);
  std::string output_layout(argv[5]);
  int64_t m = atoi(argv[6]);
  int64_t n = atoi(argv[7]);
  int64_t k = atoi(argv[8]);
  std::string kernel_name = "";
  if (argc == 10) {
    kernel_name = argv[9];
  }

  LayoutEnum layout_a = str_to_layout_enum(lhs_layout);
  if (layout_a == LayoutEnum::Invalid) {
    fprintf(stderr, "unknown lhs layout: %s\n", lhs_layout.c_str());
    return -1;
  }
  LayoutEnum layout_b = str_to_layout_enum(rhs_layout);
  if (layout_b == LayoutEnum::Invalid) {
    fprintf(stderr, "unknown rhs layout: %s\n", rhs_layout.c_str());
    return -1;
  }
  LayoutEnum layout_c = str_to_layout_enum(output_layout);
  if (layout_c == LayoutEnum::Invalid) {
    fprintf(stderr, "unknown rhs layout: %s\n", output_layout.c_str());
    return -1;
  }

  Manifest manifest;
  cutlass::library::initialize_all_gemm_operations(manifest);

  if (output_type == "fp32" && accmu_type == "fp32") {
    profile_gemm<__half, __half, float, float>(
        manifest, m, n, k, layout_a, layout_b, layout_c, EpilogueEnum::Sigmoid,
        kernel_name);
  } else if (output_type == "fp16" && accmu_type == "fp32") {
    profile_gemm<__half, __half, __half, float>(
        manifest, m, n, k, layout_a, layout_b, layout_c, EpilogueEnum::Sigmoid,
        kernel_name);
  } else if (output_type == "fp16" && accmu_type == "fp16") {
    profile_gemm<__half, __half, __half, __half>(
        manifest, m, n, k, layout_a, layout_b, layout_c, EpilogueEnum::Sigmoid,
        kernel_name);
  } else {
    fprintf(stderr, "unsupported output and accmulator type\n");
    return -1;
  }

  return 0;
}
