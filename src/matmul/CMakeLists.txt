add_executable(cublas_matmul "${CMAKE_CURRENT_SOURCE_DIR}/cublas_matmul.cpp")
# target_link_directories(cublas_matmul PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(cublas_matmul bench ${bench_CUDA_LIBRARIES})

add_executable(gemm "${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp")
# target_link_directories(gemm PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(gemm bench cutlass_gemm_library ${bench_CUDA_LIBRARIES})

add_executable(gemm_bias "${CMAKE_CURRENT_SOURCE_DIR}/gemm_bias.cpp")
# target_link_directories(gemm_bias PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(gemm_bias bench cutlass_gemm_bias_library ${bench_CUDA_LIBRARIES})

add_executable(gemm_bias_relu "${CMAKE_CURRENT_SOURCE_DIR}/gemm_bias_relu.cpp")
# target_link_directories(gemm_bias_relu PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(gemm_bias_relu bench cutlass_gemm_bias_relu_library ${bench_CUDA_LIBRARIES})

add_executable(gemm_gelu "${CMAKE_CURRENT_SOURCE_DIR}/gemm_gelu.cpp")
target_link_libraries(gemm_gelu bench cutlass_gemm_gelu_library ${bench_CUDA_LIBRARIES})

add_executable(gemm_sigmoid "${CMAKE_CURRENT_SOURCE_DIR}/gemm_sigmoid.cpp")
target_link_libraries(gemm_sigmoid bench cutlass_gemm_sigmoid_library ${bench_CUDA_LIBRARIES})


add_executable(test "${CMAKE_CURRENT_SOURCE_DIR}/test.cu")
target_link_libraries(test bench ${bench_CUDA_LIBRARIES})