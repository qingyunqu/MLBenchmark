#pragma once

#include <cuda_fp16.h>

#include "./dtype.h"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/tensor_ref.h"

// cutlass_type_to_ctype
template <typename T> struct cutlass_type_to_ctype {};
template <> struct cutlass_type_to_ctype<cutlass::half_t> {
  using type = __half;
};
template <> struct cutlass_type_to_ctype<float> { using type = float; };

// ctype_to_cutlass_type
template <typename T> struct ctype_to_cutlass_type {};
template <> struct ctype_to_cutlass_type<__half> {
  using type = cutlass::half_t;
};
template <> struct ctype_to_cutlass_type<float> { using type = float; };

// cutlass_type_to_dtype
template <typename T> struct cutlass_type_to_dtype {};
template <> struct cutlass_type_to_dtype<cutlass::half_t> {
  static constexpr DTypeEnum value = DTypeEnum::Float16;
};
template <> struct cutlass_type_to_dtype<float> {
  static constexpr DTypeEnum value = DTypeEnum::Float32;
};

template <typename T>
inline constexpr DTypeEnum cutlass_type_to_dtype_v =
    cutlass_type_to_dtype<T>::value;

// cutlass_layout_to_layout
template <typename T> struct cutlass_layout_to_layout {};
template <> struct cutlass_layout_to_layout<cutlass::layout::ColumnMajor> {
  static constexpr LayoutEnum value = LayoutEnum::ColumnMajor;
};
template <> struct cutlass_layout_to_layout<cutlass::layout::RowMajor> {
  static constexpr LayoutEnum value = LayoutEnum::RowMajor;
};
template <> struct cutlass_layout_to_layout<cutlass::layout::TensorNHWC> {
  static constexpr LayoutEnum value = LayoutEnum::NHWC;
};

template <typename T>
inline constexpr LayoutEnum cutlass_layout_to_layout_v =
    cutlass_layout_to_layout<T>::value;
