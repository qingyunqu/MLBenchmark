#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "benchmark.h"
#include "check.h"
#include "cutlass_dtype.h"
#include "matmul/check_matmul.h"
#include "matmul/cublas_matmul.h"
#include "util.h"
#include <memory>

using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations

using TA = cutlass_type_to_ctype<ElementInputA>::type;
using TB = cutlass_type_to_ctype<ElementInputB>::type;
using TO = cutlass_type_to_ctype<ElementOutput>::type;
using CompOn = cutlass_type_to_ctype<ElementAccumulator>::type;

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    4,  // <- this is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

// using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
//                                          LayoutInputA,
//                                          ElementInputB,
//                                          LayoutInputB,
//                                          ElementOutput,
//                                          LayoutOutput,
//                                          ElementAccumulator,
//                                          MMAOp,
//                                          SmArch,
//                                          ShapeMMAThreadBlock,
//                                          ShapeMMAWarp,
//                                          ShapeMMAOp,
//                                          EpilogueOp,
//                                          SwizzleThreadBlock,
//                                          NumStages>;

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
    
  >;

int run() {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  const int m = 4096;
  const int n = 4096;
  const int k = 4096;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.mn());

  RandCPUBuffer<TA>((TA*)tensor_a.host_view().data(), m * k);
  RandCPUBuffer<TB>((TB*)tensor_b.host_view().data(), k * n);
  RandCPUBuffer<TO>((TO*)tensor_c.host_view().data(), m * n);
  // FillCPUBuffer<TO>((TO*)tensor_d.host_view().data(), m * n);
  // FillCPUBuffer<TO>((TO*)tensor_ref_d.host_view().data(), m * n);
  RandCPUBuffer<TO>((TO*)tensor_d.host_view().data(), m * n);
  RandCPUBuffer<TO>((TO*)tensor_ref_d.host_view().data(), m * n);

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    printf("gemm_op.can_implement() failed\n");
  }
  CUTLASS_CHECK(status);

  cudaStream_t stream = nullptr;
  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get(), stream);
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  CUTLASS_CHECK(gemm_op());

  float time = benchmark_cutlass<Gemm>(&gemm_op, stream);
  printf("time: %f ms\n", time);

  // Copy output data from CUTLASS and reference kernel to host for comparison
  // tensor_d.sync_host();
  // tensor_ref_d.sync_host();

  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  Matmul<TA, TO>* op = new CublasMatmul<TA, TO, CompOn>((int64_t)m, (int64_t)n, (int64_t)k, true, true, false, handle);
  op->Run((TA*)tensor_a.device_ref().data(), (TB*)tensor_b.device_ref().data(), (TO*)tensor_ref_d.device_ref().data());
  CUDACHECK(cudaDeviceSynchronize());
  
  bool passed = CheckCUDABuffer<TO>((TO*)tensor_d.device_ref().data(), (TO*)tensor_ref_d.device_ref().data(), m * n, 1e-5f);
  std::cout << (passed ? "Passed" : "Failed") << std::endl;
  
  time = benchmark<TA, TO>(op, stream, (TA*)tensor_a.device_ref().data(), (TB*)tensor_b.device_ref().data(), (TO*)tensor_ref_d.device_ref().data());
  printf("cublas time: %f ms\n", time);
  CUBLASCHECK(cublasDestroy(handle));
  delete op;
  

  // PrintCUDABuffer<TO>((TO*)tensor_d.device_ref().data(), 20);
  // PrintCUDABuffer<TO>((TO*)tensor_ref_d.device_ref().data(), 20);

  return (passed ? 0  : -1);
}

int main() {

  // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
  //
  // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

    // Returning zero when built on older Toolkits so tests pass. The actions of this SDK example are no-op.
    return 0;
  }
  else {
    return run();
  }
}

