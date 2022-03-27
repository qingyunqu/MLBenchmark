add_executable(fuse_gemm_gemm "${CMAKE_CURRENT_SOURCE_DIR}/fuse_gemm_gemm.cu")
target_include_directories(fuse_gemm_gemm PRIVATE "${REPO_ROOT}/include/b2b")
target_link_libraries(fuse_gemm_gemm bench ${bench_CUDA_LIBRARIES})