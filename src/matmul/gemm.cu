#include "Manifest.h"
#include "profile.h"

#include <cassert>
#include <cuda_fp16.h>
#include <string>

using Operation_cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nn_align8 =
    cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t, 8, float, float,
            cutlass::epilogue::thread::ScaleType::Default>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3, 8, 8,
        false, cutlass::arch::OpMultiplyAdd

        >;

namespace cutlass {
namespace library {
void initialize_all_gemm_operations(Manifest &manifest);
void initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8(
    Manifest &manifest);
} // namespace library
} // namespace cutlass

int main(int argc, char *argv[]) {
  assert(argc == 9);
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
  cutlass::library::
      initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8(
          manifest);

  if (output_type == "fp32" && accmu_type == "fp32") {
    profile_gemm<__half, __half, float, float>(manifest, m, n, k, layout_a,
                                               layout_b, layout_c,
                                               EpilogueEnum::None, {});
  } else if (output_type == "fp16" && accmu_type == "fp32") {
    profile_gemm<__half, __half, __half, float>(manifest, m, n, k, layout_a,
                                                layout_b, layout_c,
                                                EpilogueEnum::None, {});
  } else if (output_type == "fp16" && accmu_type == "fp16") {
    profile_gemm<__half, __half, __half, __half>(manifest, m, n, k, layout_a,
                                                 layout_b, layout_c,
                                                 EpilogueEnum::None, {});
  } else {
    fprintf(stderr, "unsupported output and accmulator type\n");
    return -1;
  }

  return 0;
}