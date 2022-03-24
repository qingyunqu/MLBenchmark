add_executable(cudnn_conv "${CMAKE_CURRENT_SOURCE_DIR}/cudnn_conv.cpp")
target_link_directories(cudnn_conv PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(cudnn_conv bench ${bench_CUDA_LIBRARIES})

add_executable(conv2d "${REPO_ROOT}/src/convolution/conv2d.cu")
target_link_directories(conv2d PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(conv2d bench cutlass_conv2d_library ${bench_CUDA_LIBRARIES})
