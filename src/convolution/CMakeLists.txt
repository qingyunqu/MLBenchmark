add_executable(cudnn_conv "${CMAKE_CURRENT_SOURCE_DIR}/cudnn_conv.cpp")
target_link_directories(cudnn_conv PRIVATE ${bench_CUDA_LINK_DIRS})
target_link_libraries(cudnn_conv bench ${bench_CUDA_LIBRARIES})
