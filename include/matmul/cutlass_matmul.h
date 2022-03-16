#pragma once

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "../check.h"
#include "../ops.h"

template <typename T> class CutlassMatmulWrap {
public:
  using RowMajor = cutlass::layout::RowMajor;
  using MMAOp = cutlass::arch::OpClassWmmaTensorOp;
  using Sm = cutlass::arch::Sm75;
  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  const int split_k = 1;
  using Gemm =
      cutlass::gemm::device::Gemm<cutlass::half_t, RowMajor, cutlass::half_t,
                                  RowMajor, cutlass::half_t, RowMajor,
                                  cutlass::half_t, MMAOp, Sm, ThreadBlockShape,
                                  WarpShape, InstructionShape>;

  CutlassMatmulWrap() = default;
  void Run(T *a_val, T *b_val, T *c_val, int64_t M, int64_t N, int64_t K,
           cudaStream_t stream) {
    typename Gemm::Arguments args(
        cutlass::gemm::GemmCoord{M, N, K},
        cutlass::TensorRef<cutlass::half_t, RowMajor>{(cutlass::half_t *)a_val,
                                                      K},
        cutlass::TensorRef<cutlass::half_t, RowMajor>{(cutlass::half_t *)b_val,
                                                      N},
        cutlass::TensorRef<cutlass::half_t, RowMajor>{c_val, N},
        cutlass::TensorRef<cutlass::half_t, RowMajor>{c_val, N}, {1.f, 0.f},
        split_k);
    CUTLASS_CHECK(gemm(args, nullptr, stream));
  }

private:
  Gemm gemm;
};

template <typename T, typename CompOn = float>
class CutlassMatmul : public Matmul<T> {
public:
  CutlassMatmul(bool lhs_transpose, bool rhs_transpose, bool output_transpose,
                int64_t m, int64_t n, int64_t k, cudaStream_t stream)
      : Matmul<T>(m, n, k, lhs_transpose, rhs_transpose, output_transpose),
        stream(stream) {}
  virtual void Run(const T *a_val, const T *b_val, T *c_val) override {
    wrap.Run(const_cast<T *>(a_val), const_cast<T *>(b_val), c_val, m, n, k,
             stream);
  }
  virtual ~CutlassMatmul() = default;

private:
  CutlassMatmulWrap<T> wrap;
  cudaStream_t stream;
};