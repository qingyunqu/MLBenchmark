#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "../check.h"
#include "../ops.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// Gemm operator cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8
using Operation_cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8 =
    cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2, 8, 8,
        false, cutlass::arch::OpMultiplyAdd

        >;

///////////////////////////////////////////////////////////////////////////////////////////////////

class Operation {
public:
  virtual void Run() = 0;
};

template <typename Gemm> class MatmulOperation : public Operation {
public:
  MatmulOperation(const char *kernel_name) : kernel_name(kernel_name) {}

  virtual void Run() {}

private:
  const char *kernel_name;
  Gemm gemm;
};

namespace matmul {

class Manifest {
public:
  template <typename ElementInputA, typename LayoutInputA,
            typename ElementInputB, typename LayoutInputB,
            typename ElementOutput, typename LayoutOutput,
            typename ElementAccumulator>
  void append(Operation *op);

private:
  std::vector<Operation *> hhhs_nnt;
};

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::ColumnMajor, float,
                      cutlass::layout::RowMajor, float>(Operation *op) {
  hhhs_nnt.push_back(op);
}

} // namesapce matmul