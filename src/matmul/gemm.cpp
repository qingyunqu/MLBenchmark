#include "Manifest.h"
#include "profile.h"

#include <cassert>
#include <cuda_fp16.h>
#include <string>
#include <unordered_set>
#include <regex>
#include <iostream>

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main(int argc, char *argv[]) {
  assert(argc >= 9);
  std::string output_type(argv[1]);
  std::string accmu_type(argv[2]);
  std::string lhs_layout(argv[3]);
  std::string rhs_layout(argv[4]);
  std::string output_layout(argv[5]);
  int64_t m = atoi(argv[6]);
  int64_t n = atoi(argv[7]);
  int64_t k = atoi(argv[8]);

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

  std::unordered_set<std::string> run_kernels;
  if (argc >= 10) {
    for (size_t i = 9; i < argc; i++) {
      std::string argument(argv[i]);
      if (argument[0] == '"') {
        argument = argument.substr(1);
      }
      if (argument[argument.size() - 1] == '"') {
        argument = argument.substr(0, argument.size() - 1);
      }
      std::cout << "regex: " << argument << "\n";
      std::regex reg(argument);
      for (auto kernel : manifest.kernels) {
        if (std::regex_match(kernel->Name(), reg)) {
          run_kernels.insert(kernel->Name());
        }
      }
    }
  }

  if (output_type == "fp32" && accmu_type == "fp32") {
    profile_gemm<__half, __half, float, float>(manifest, m, n, k, layout_a,
                                               layout_b, layout_c,
                                               EpilogueEnum::None, run_kernels);
  } else if (output_type == "fp16" && accmu_type == "fp32") {
    profile_gemm<__half, __half, __half, float>(
        manifest, m, n, k, layout_a, layout_b, layout_c, EpilogueEnum::None,
        run_kernels);
  } else if (output_type == "fp16" && accmu_type == "fp16") {
    profile_gemm<__half, __half, __half, __half>(
        manifest, m, n, k, layout_a, layout_b, layout_c, EpilogueEnum::None,
        run_kernels);
  } else {
    fprintf(stderr, "unsupported output and accmulator type\n");
    return -1;
  }

  return 0;
}