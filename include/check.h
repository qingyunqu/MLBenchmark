#pragma once

#define CUDACHECK(cmd)                                                    \
    do {                                                                  \
        cudaError_t e = cmd;                                              \
        if (e != cudaSuccess) {                                           \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NCCLCHECK(cmd)                                                    \
    do {                                                                  \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess) {                                           \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUDNNCHECK(expression)                                     \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS) {                      \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

#define CUBLASCHECK(expression)                                     \
    {                                                              \
        cublasStatus_t status = (expression);                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                      \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cublasGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define after_kernel_launch()           \
    do {                                \
        CUDACHECK(cudaGetLastError());  \
    } while (0)
