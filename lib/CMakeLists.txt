file(GLOB_RECURSE bench_src CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/**/*.cpp")

add_library(bench STATIC ${bench_src})
set_target_properties(bench PROPERTIES CUDA_ARCHITECTURES "75;80")
