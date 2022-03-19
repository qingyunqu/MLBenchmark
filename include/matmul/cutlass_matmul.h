#pragma once

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "../check.h"
#include "../ops.h"

// template <typename T, typename To> class CutlassMatmulWrap {
// public:
//   using ElementAccumulator = float;
//   using ElementComputeEpilogue = ElementAccumulator;
//   using ElementInputA = cutlass::half_t;
//   using ElementInputB = cutlass::half_t;
//   using ElementOutput = cutlass::half_t;
//   using LayoutInputA = cutlass::layout::RowMajor;
//   using LayoutInputB = cutlass::layout::RowMajor;
//   using LayoutOutput = cutlass::layout::RowMajor;
//   using MMAOp = cutlass::arch::OpClassTensorOp;
//   using SmArch = cutlass::arch::Sm70;
//   using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
//   using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
//   using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
//   using SwizzleThreadBlock =
//       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
//       ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
//       ElementAccumulator, ElementComputeEpilogue>;
//   constexpr int NumStages = 2;

//   using Gemm = cutlass::gemm::device::Gemm<
//       ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
//       ElementOutput,
//       LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
//       ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

//   CutlassMatmulWrap() = default;
//   void Run(T *a_val, T *b_val, To *c_val, int64_t M, int64_t N, int64_t K,
//            cudaStream_t stream) {
//     int split_k = 1;
//     typename Gemm::Arguments args(
//         cutlass::gemm::GemmCoord{M, N, K},
//         cutlass::TensorRef<cutlass::half_t, LayoutInputA>{
//             (cutlass::half_t *)a_val, K},
//         cutlass::TensorRef<cutlass::half_t, LayoutInputB>{
//             (cutlass::half_t *)b_val, N},
//         cutlass::TensorRef<cutlass::half_t, LayoutOutput>{c_val, N},
//         cutlass::TensorRef<cutlass::half_t, LayoutOutput>{c_val, N}, {1.f,
//         0.f},
//         split_k);
//     CUTLASS_CHECK(gemm(args, nullptr, stream));
//   }

// private:
//   Gemm gemm;
// };

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue>;
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

template <typename T, typename To, typename CompOn = float>
class CutlassMatmul : public Matmul<T, To> {
public:
  CutlassMatmul(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
                bool rhs_transpose, bool output_transpose, cudaStream_t stream)
      : Matmul<T, To>(m, n, k, lhs_transpose, rhs_transpose, output_transpose),
        stream(stream) {}
  virtual bool Check() override { return true; }
  virtual void Run(const T *a_val, const T *b_val, To *c_val) override {
    Gemm gemm;
    int split_k = 1;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmCoord{m, n, k},
        cutlass::TensorRef<cutlass::half_t, LayoutInputA>{
            (cutlass::half_t *)a_val, k},
        cutlass::TensorRef<cutlass::half_t, LayoutInputB>{
            (cutlass::half_t *)b_val, n},
        cutlass::TensorRef<cutlass::half_t, LayoutOutput>{c_val, n},
        cutlass::TensorRef<cutlass::half_t, LayoutOutput>{c_val, n}, {1.f, 0.f},
        split_k);
    CUTLASS_CHECK(gemm(args, nullptr, stream));
    // wrap.Run(const_cast<T *>(a_val), const_cast<T *>(b_val), c_val, m, n, k,
    //          stream);
  }
  virtual ~CutlassMatmul() = default;

private:
  // CutlassMatmulWrap<T, To> wrap;
  cudaStream_t stream;
};