# Minimum CMake required
cmake_minimum_required(VERSION 3.18)

# Project
project(bench C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-w)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
enable_language(CUDA)
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA_HOME = ${CUDA_TOOLKIT_ROOT_DIR}")

include_directories("${CUDA_INCLUDE_DIRS}")
set(bench_CUDA_LIBRARIES ${CUDA_LIBRARIES})
list(APPEND bench_CUDA_LIBRARIES cublas cudnn)
# set(bench_CUDA_LINK_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

set(REPO_ROOT ${PROJECT_SOURCE_DIR})
message("REPO_ROOT = ${REPO_ROOT}")
set(GEN_ROOT ${REPO_ROOT}/kernels)
message("GEN_ROOT = ${GEN_ROOT}")
set(CUTLASS_ROOT ${REPO_ROOT}/../cutlass)
message("CUTLASS_ROOT = ${CUTLASS_ROOT}")

include_directories("${REPO_ROOT}/include")
include_directories("${CUTLASS_ROOT}/include")
include_directories("${CUTLASS_ROOT}/tools/util/include")

# bench lib
add_subdirectory(${REPO_ROOT}/lib)

# generated kernels
add_subdirectory(${GEN_ROOT})

# convolution
add_subdirectory(${REPO_ROOT}/src/convolution)

# matmul
add_subdirectory(${REPO_ROOT}/src/matmul)
