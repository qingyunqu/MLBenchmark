#include "Manifest.h"
#include "profile.h"

#include <cuda_fp16.h>

namespace cutlass {
namespace library {
void initialize_all_conv2d_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main(int argc, char *argv[]) {
  assert(argc == 12 || argc == 13);
  int64_t iH = atoi(argv[1]);
  int64_t iW = atoi(argv[2]);
  int64_t iC = atoi(argv[3]);
  int64_t oC = atoi(argv[4]);
  int64_t kH = atoi(argv[5]);
  int64_t kW = atoi(argv[6]);
  int64_t strideH = atoi(argv[7]);
  int64_t strideW = atoi(argv[8]);
  int64_t paddingH = atoi(argv[9]);
  int64_t paddingW = atoi(argv[10]);
  int64_t N = atoi(argv[11]);
  std::string kernel_name = "";
  if (argc == 13) {
    kernel_name = argv[12];
  }

  int64_t dilationH = 1;
  int64_t dilationW = 1;
  int64_t oH = (iH + 2 * paddingH - dilationH * (kH - 1) - 1) / strideH + 1;
  int64_t oW = (iW + 2 * paddingW - dilationW * (kW - 1) - 1) / strideW + 1;

  Manifest manifest;
  cutlass::library::initialize_all_conv2d_operations(manifest);

  profile_conv2d<__half, __half, __half, float>(
      manifest, N, iH, iW, iC, oH, oW, oC, kH, kW, strideH, strideW, paddingH,
      paddingW, dilationH, dilationW, EpilogueEnum::Grelu, kernel_name);
  return 0;
}
