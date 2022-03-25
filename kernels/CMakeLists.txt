# cutlass generated conv2d
file(GLOB_RECURSE cutlass_conv2d_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conv2d/*.cu")
add_library(cutlass_conv2d_library ${cutlass_conv2d_src})

# cutlass generated gemm
file(GLOB_RECURSE cutlass_gemm_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.cu")
add_library(cutlass_gemm_library ${cutlass_gemm_src})

# cutlass generated gemm_bias
file(GLOB_RECURSE cutlass_gemm_bias_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm_bias/*.cu")
add_library(cutlass_gemm_bias_library ${cutlass_gemm_bias_src})