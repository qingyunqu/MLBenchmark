# cutlass generated conv2d
file(GLOB_RECURSE cutlass_conv2d_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conv2d/*.cu")
add_library(cutlass_conv2d_library ${cutlass_conv2d_src})

# cutlass generated conv2d_sm80
file(GLOB_RECURSE cutlass_conv2d_sm80_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_sm80/*.cu")
add_library(cutlass_conv2d_sm80_library ${cutlass_conv2d_sm80_src})
set_target_properties(cutlass_conv2d_sm80_library PROPERTIES CUDA_ARCHITECTURES "80")

# cutlass generated conv2d_bias
file(GLOB_RECURSE cutlass_conv2d_bias_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias/*.cu")
add_library(cutlass_conv2d_bias_library ${cutlass_conv2d_bias_src})

# cutlass generated conv2d_bias_relu
file(GLOB_RECURSE cutlass_conv2d_bias_relu_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_relu/*.cu")
add_library(cutlass_conv2d_bias_relu_library ${cutlass_conv2d_bias_relu_src})

# cutlass generated conv2d_bias_sigmoid
file(GLOB_RECURSE cutlass_conv2d_bias_sigmoid_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_sigmoid/*.cu")
add_library(cutlass_conv2d_bias_sigmoid_library ${cutlass_conv2d_bias_sigmoid_src})

# cutlass generated conv2d_gelu
file(GLOB_RECURSE cutlass_conv2d_gelu_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_gelu/*.cu")
add_library(cutlass_conv2d_gelu_library ${cutlass_conv2d_gelu_src})





# cutlass generated gemm
file(GLOB_RECURSE cutlass_gemm_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.cu")
add_library(cutlass_gemm_library ${cutlass_gemm_src})

# cutlass generated gemm_bias
file(GLOB_RECURSE cutlass_gemm_bias_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm_bias/*.cu")
add_library(cutlass_gemm_bias_library ${cutlass_gemm_bias_src})

# cutlass generated gemm_bias_relu
file(GLOB_RECURSE cutlass_gemm_bias_relu_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm_bias_relu/*.cu")
add_library(cutlass_gemm_bias_relu_library ${cutlass_gemm_bias_relu_src})

# cutlass generated gemm_gelu
file(GLOB_RECURSE cutlass_gemm_gelu_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm_gelu/*.cu")
add_library(cutlass_gemm_gelu_library ${cutlass_gemm_gelu_src})

# cutlass generated gemm_sigmoid
file(GLOB_RECURSE cutlass_gemm_sigmoid_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm_sigmoid/*.cu")
add_library(cutlass_gemm_sigmoid_library ${cutlass_gemm_sigmoid_src})