# Minimum CMake required
cmake_minimum_required(VERSION 3.18)

# Project
project(brt C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# linking 
# set(brt_LINK_DIRS )

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
# list(APPEND brt_LINK_DIRS ${brt_CUDNN_HOME}/lib64 ${brt_CUDA_HOME}/lib64)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
enable_language(CUDA)
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

set(BRT_CUDA_LIBRARIES ${CUDA_LIBRARIES})
list(APPEND BRT_CUDA_LIBRARIES cublas cudnn)


set(REPO_ROOT ${PROJECT_SOURCE_DIR})
message("REPO_ROOT = ${REPO_ROOT}")
set(GEN_ROOT ${REPO_ROOT}/generated)
message("GEN_ROOT = ${GEN_ROOT}")
set(CUTLASS_ROOT ${REPO_ROOT}/../cutlass)
message("CUTLASS_ROOT = ${CUTLASS_ROOT}")

include_directories("${REPO_ROOT}/include")
include_directories("${CUTLASS_ROOT}/include")
include_directories("${CUTLASS_ROOT}/tools/util/include")

add_executable(gemm "${REPO_ROOT}/src/matmul/gemm.cu")
target_link_libraries(gemm BRT_CUDA_LIBRARIES)
