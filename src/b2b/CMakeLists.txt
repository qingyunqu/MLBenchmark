add_executable(b2b_gemm "${CMAKE_CURRENT_SOURCE_DIR}/b2b_gemm.cu")
target_include_directories(b2b_gemm PRIVATE "${REPO_ROOT}/include/b2b")
target_link_libraries(b2b_gemm bench cutlass_gemm_library ${bench_CUDA_LIBRARIES})

add_executable(b2b_conv "${CMAKE_CURRENT_SOURCE_DIR}/b2b_conv.cu")
target_include_directories(b2b_conv PRIVATE "${REPO_ROOT}/include/b2b")
target_link_libraries(b2b_conv bench ${bench_CUDA_LIBRARIES})
