# Minimum CMake required
cmake_minimum_required(VERSION 3.18)

# Project
project(bench C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
enable_language(CUDA)
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

set(bench_CUDA_LIBRARIES ${CUDA_LIBRARIES})
list(APPEND bench_CUDA_LIBRARIES cublas cudnn)
message("CUDA_HOME = ${CUDA_TOOLKIT_ROOT_DIR}")
set(bench_CUDA_LINK_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(REPO_ROOT ${PROJECT_SOURCE_DIR})
message("REPO_ROOT = ${REPO_ROOT}")
set(GEN_ROOT ${REPO_ROOT}/generated)
message("GEN_ROOT = ${GEN_ROOT}")
set(CUTLASS_ROOT ${REPO_ROOT}/../cutlass)
message("CUTLASS_ROOT = ${CUTLASS_ROOT}")

include_directories("${REPO_ROOT}/include")
include_directories("${CUTLASS_ROOT}/include")
include_directories("${CUTLASS_ROOT}/tools/util/include")

# lib src
file(GLOB_RECURSE bench_src CONFIGURE_DEPENDS 
    "${REPO_ROOT}/lib/*.cpp"
    "${REPO_ROOT}/lib/**/*.cpp")
add_library(bench STATIC ${bench_src})

add_subdirectory(${REPO_ROOT}/src/convolution)

# cutlass generated
# file(GLOB_RECURSE cutlass_gemm_src CONFIGURE_DEPENDS "${GEN_ROOT}/gemm/*.cu")
# add_library(cutlass_gemm_library ${cutlass_gemm_src})

# add_executable(gemm "${REPO_ROOT}/src/matmul/gemm.cu")
# target_link_libraries(gemm bench_CUDA_LIBRARIES cutlass_gemm_library)
