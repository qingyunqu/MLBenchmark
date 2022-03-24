add_executable(cublas_matmul "${CMAKE_CURRENT_SOURCE_DIR}/cublas_matmul.cpp")
target_link_directories(cublas_matmul PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(cublas_matmul bench ${bench_CUDA_LIBRARIES})

