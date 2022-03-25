add_executable(cudnn_conv "${CMAKE_CURRENT_SOURCE_DIR}/cudnn_conv.cpp")
target_link_directories(cudnn_conv PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(cudnn_conv bench ${bench_CUDA_LIBRARIES})

add_executable(conv2d "${CMAKE_CURRENT_SOURCE_DIR}/conv2d.cpp")
target_link_directories(conv2d PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(conv2d bench cutlass_conv2d_library ${bench_CUDA_LIBRARIES})

add_executable(conv2d_bias "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias.cpp")
target_link_directories(conv2d_bias PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(conv2d_bias bench cutlass_conv2d_bias_library ${bench_CUDA_LIBRARIES})

add_executable(conv2d_bias_relu "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_relu.cpp")
target_link_directories(conv2d_bias_relu PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(conv2d_bias_relu bench cutlass_conv2d_bias_relu_library ${bench_CUDA_LIBRARIES})