#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"

#include "../check.h"
#include "../cutlass_dtype.h"
#include "../ops.h"
#include "../util.h"
#include "./cublas_matmul.h"

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
  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c) = 0;
  virtual bool Check() = 0;
  virtual void Initialize(cudaStream_t) = 0;
  virtual void Run() = 0;
  virtual const char *Name() = 0;
};

template <typename Gemm> class MatmulOperation : public Operation {
public:
  using ElementA = typename Gemm::ElementA;
  using LayoutA = typename Gemm::LayoutA;
  using ElementB = typename Gemm::ElementB;
  using LayoutB = typename Gemm::LayoutB;
  using ElementC = typename Gemm::ElementC;
  using LayoutC = typename Gemm::LayoutC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  MatmulOperation(const char *kernel_name) : kernel_name(kernel_name) {}

  virtual void SetArgument(int64_t m, int64_t n, int64_t k, void *a, void *b,
                           void *c) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    bool lhs_transpose = cutlass_matrix_transpose<LayoutA>::value;
    bool rhs_transpose = cutlass_matrix_transpose<LayoutB>::value;
    bool output_transpose = cutlass_matrix_transpose<LayoutC>::value;
    LayoutA layoutA(k);
    LayoutB layoutB(n);
    LayoutC layoutC(n);
    if (lhs_transpose) {
      layoutA = LayoutA(m);
    }
    if (rhs_transpose) {
      layoutB = LayoutB(k);
    }
    if (output_transpose) {
      layoutC = LayoutC(m);
    }
    arguments = {problem_size,
                 {(ElementA *)a, layoutA},
                 {(ElementB *)b, layoutB},
                 {(ElementC *)c, layoutC},
                 {(ElementC *)c, layoutC},
                 {(ElementAccumulator)1, (ElementAccumulator)0},
                 /*split_k_slices=*/1};
  }
  virtual bool Check() {
    return gemm.can_implement(arguments) == cutlass::Status::kSuccess;
  }
  virtual void Initialize(cudaStream_t stream) {
    CUTLASS_CHECK(gemm.initialize(arguments, nullptr, stream));
  }
  virtual void Run() { CUTLASS_CHECK(gemm()); }
  virtual const char *Name() { return kernel_name; }

private:
  const char *kernel_name;
  Gemm gemm;
  typename Gemm::Arguments arguments;
};

namespace matmul {

class Manifest {
public:
  template <typename ElementInputA, typename LayoutInputA,
            typename ElementInputB, typename LayoutInputB,
            typename ElementOutput, typename LayoutOutput,
            typename ElementAccumulator>
  void append(Operation *op);

  template <typename TA, typename TB, typename TC>
  void init_tensor(int64_t m, int64_t n, int64_t k, TA *&a, TB *&b, TC *&c,
                   TC *&ref_c) {
    CUDACHECK(cudaMalloc(&a, m * k * sizeof(TA)));
    CUDACHECK(cudaMalloc(&b, n * k * sizeof(TB)));
    CUDACHECK(cudaMalloc(&c, m * n * sizeof(TC)));
    CUDACHECK(cudaMalloc(&ref_c, m * n * sizeof(TC)));
    RandCUDABuffer(a, m * k);
    RandCUDABuffer(b, n * k);
    RandCUDABuffer(c, m * n);
    FillCUDABuffer(ref_c, m * n);
  }

  template <typename TA, typename TB, typename TC, typename ElementAccumulator>
  void profile_interal(std::vector<Operation *> &list, int64_t m, int64_t n,
                       int64_t k, bool lhs_transpose, bool rhs_transpose,
                       bool output_transpose, TA *a, TB *b, TC *c, TC *ref_c) {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    CUBLASCHECK(cublasSetStream(handle, stream));
    Matmul<TA, TC> *op = new CublasMatmul<TA, TC, ElementAccumulator>(
        m, n, k, lhs_transpose, rhs_transpose, output_transpose, handle);
    op->Run(a, b, ref_c);
    CUDACHECK(cudaDeviceSynchronize());
    CUBLASCHECK(cublasDestroy(handle));
    delete op;

    for (auto &op : list) {
      op->SetArgument(m, n, k, (void *)a, (void *)b, (void *)c);
      if (!op->Check()) {
        continue;
      }
      op->Initialize(stream);
      op->Run();
      bool passed = CheckCUDABuffer<TC>(c, ref_c, m * n, 1e-5f);
      std::cout << op->Name() << " : " << (passed ? "Passed" : "Failed")
                << std::endl;
    }
  }

  template <typename ElementInputA, typename ElementInputB,
            typename ElementOutput, typename ElementAccumulator>
  void profile(int64_t m, int64_t n, int64_t k, bool lhs_transpose,
               bool rhs_transpose, bool output_transpose);

private:
  std::vector<Operation *> hhss_nnt;
  std::vector<Operation *> hhss_ntt;
  std::vector<Operation *> hhss_tnt;
  std::vector<Operation *> hhss_ttt;
  std::vector<Operation *> hhhs_nnt;
  std::vector<Operation *> hhhs_ntt;
  std::vector<Operation *> hhhs_tnt;
  std::vector<Operation *> hhhs_ttt;
  std::vector<Operation *> hhhh_nnt;
  std::vector<Operation *> hhhh_ntt;
  std::vector<Operation *> hhhh_tnt;
  std::vector<Operation *> hhhh_ttt;
};

// hhss
template <>
void Manifest::profile<cutlass::half_t, cutlass::half_t, float, float>(
    int64_t m, int64_t n, int64_t k, bool lhs_transpose, bool rhs_transpose,
    bool output_transpose) {
  __half *a = nullptr;
  __half *b = nullptr;
  float *c = nullptr;
  float *ref_c = nullptr;
  init_tensor<__half, __half, float>(m, n, k, a, b, c, ref_c);
  if (lhs_transpose && rhs_transpose) {
    profile_interal<__half, __half, float, float>(
        hhss_nnt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else if (lhs_transpose && !rhs_transpose) {
    profile_interal<__half, __half, float, float>(
        hhss_ntt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else if (!lhs_transpose && rhs_transpose) {
    profile_interal<__half, __half, float, float>(
        hhss_tnt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else {
    profile_interal<__half, __half, float, float>(
        hhss_ttt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  }
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::ColumnMajor, float,
                      cutlass::layout::RowMajor, float>(Operation *op) {
  hhss_nnt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::RowMajor, float,
                      cutlass::layout::RowMajor, float>(Operation *op) {
  hhss_ntt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::ColumnMajor, float,
                      cutlass::layout::RowMajor, float>(Operation *op) {
  hhss_tnt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::RowMajor, float,
                      cutlass::layout::RowMajor, float>(Operation *op) {
  hhss_ttt.push_back(op);
}

// hhhs
template <>
void Manifest::profile<cutlass::half_t, cutlass::half_t, cutlass::half_t,
                       float>(int64_t m, int64_t n, int64_t k,
                              bool lhs_transpose, bool rhs_transpose,
                              bool output_transpose) {
  __half *a = nullptr;
  __half *b = nullptr;
  __half *c = nullptr;
  __half *ref_c = nullptr;
  init_tensor<__half, __half, __half>(m, n, k, a, b, c, ref_c);
  if (lhs_transpose && rhs_transpose) {
    profile_interal<__half, __half, __half, float>(
        hhhs_nnt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else if (lhs_transpose && !rhs_transpose) {
    profile_interal<__half, __half, __half, float>(
        hhhs_ntt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else if (!lhs_transpose && rhs_transpose) {
    profile_interal<__half, __half, __half, float>(
        hhhs_tnt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else {
    profile_interal<__half, __half, __half, float>(
        hhhs_ttt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  }
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::RowMajor, float>(
    Operation *op) {
  hhhs_nnt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::RowMajor, float>(
    Operation *op) {
  hhhs_ntt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::RowMajor, float>(
    Operation *op) {
  hhhs_tnt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::RowMajor, float>(
    Operation *op) {
  hhhs_ttt.push_back(op);
}

// hhhh
template <>
void Manifest::profile<cutlass::half_t, cutlass::half_t, cutlass::half_t,
                       cutlass::half_t>(int64_t m, int64_t n, int64_t k,
                                        bool lhs_transpose, bool rhs_transpose,
                                        bool output_transpose) {
  __half *a = nullptr;
  __half *b = nullptr;
  __half *c = nullptr;
  __half *ref_c = nullptr;
  init_tensor<__half, __half, __half>(m, n, k, a, b, c, ref_c);
  if (lhs_transpose && rhs_transpose) {
    profile_interal<__half, __half, __half, __half>(
        hhhh_nnt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else if (lhs_transpose && !rhs_transpose) {
    profile_interal<__half, __half, __half, __half>(
        hhhh_ntt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else if (!lhs_transpose && rhs_transpose) {
    profile_interal<__half, __half, __half, __half>(
        hhhh_tnt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  } else {
    profile_interal<__half, __half, __half, __half>(
        hhhh_ttt, m, n, k, lhs_transpose, rhs_transpose, output_transpose, a, b,
        c, ref_c);
  }
  CUDACHECK(cudaFree(a));
  CUDACHECK(cudaFree(b));
  CUDACHECK(cudaFree(c));
  CUDACHECK(cudaFree(ref_c));
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t>(Operation *op) {
  hhhh_nnt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t>(Operation *op) {
  hhhh_ntt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::ColumnMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t>(Operation *op) {
  hhhh_tnt.push_back(op);
}

template <>
void Manifest::append<cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t, cutlass::layout::RowMajor,
                      cutlass::half_t>(Operation *op) {
  hhhh_ttt.push_back(op);
}

} // namesapce matmul