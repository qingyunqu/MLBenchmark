#include "Manifest.h"
#include "profile.h"

#include <cuda_fp16.h>

namespace cutlass {
namespace library {
void initialize_all_conv2d_operations(Manifest &manifest);
} // namespace library
} // namespace cutlass

int main() {
  Manifest manifest;
  cutlass::library::initialize_all_conv2d_operations(manifest);

  profile_conv2d_bias<__half, __half, __half, float>(
      manifest, 16, 56, 56, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1,
      EpilogueEnum::Sigmoid);
  profile_conv2d_bias<__half, __half, __half, float>(
      manifest, 16, 56, 56, 64, 28, 28, 128, 1, 1, 2, 2, 0, 0, 1, 1,
      EpilogueEnum::Sigmoid);
  return 0;
}
