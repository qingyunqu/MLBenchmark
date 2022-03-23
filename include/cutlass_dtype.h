#pragma once

#include "cutlass/cutlass.h"

template <typename T> struct cutlass_type_to_ctype {};
template <> struct cutlass_type_to_ctype<cutlass::half_t> {
  using type = __half;
};
template <> struct cutlass_type_to_ctype<float> { using type = float; };

template <typename T> struct ctype_to_cutlass_type {};
template <> struct ctype_to_cutlass_type<__half> {
  using type = cutlass::half_t;
};
template <> struct ctype_to_cutlass_type<float> { using type = float; };

template <typename T> struct cutlass_matrix_transpose {};
template <> struct cutlass_matrix_transpose<cutlass::layout::ColumnMajor> {
  static constexpr bool value = true;
};
template <> struct cutlass_matrix_transpose<cutlass::layout::RowMajor> {
  static constexpr bool value = false;
};
