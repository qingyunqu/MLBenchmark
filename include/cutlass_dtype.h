#pragma once

#include "cutlass/cutlass.h"

template <typename T>
struct cutlass_type_to_ctype {};

template <> struct cutlass_type_to_ctype<cutlass::half_t> {
  using type = __half;
};

template <> struct cutlass_type_to_ctype<float> {
  using type = float;
};