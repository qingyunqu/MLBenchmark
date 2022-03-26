#pragma once

#include "Manifest.h"
#include "dtype.h"

#include <string>

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm(Manifest &manifest, int64_t m, int64_t n, int64_t k,
                  LayoutEnum layout_a, LayoutEnum layout_b,
                  LayoutEnum layout_c);

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_gemm_bias(Manifest &manifest, int64_t m, int64_t n, int64_t k,
                       LayoutEnum layout_a, LayoutEnum layout_b,
                       LayoutEnum layout_c,
                       EpilogueEnum epilogue = EpilogueEnum::None);

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d(Manifest &manifest, int64_t N, int64_t iH, int64_t iW,
                    int64_t iC, int64_t oH, int64_t oW, int64_t oC, int64_t kH,
                    int64_t kW, int64_t strideH, int64_t strideW,
                    int64_t paddingH, int64_t paddingW, int64_t dilationH = 1,
                    int64_t dilationW = 1);

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d_bias(Manifest &manifest, int64_t N, int64_t iH, int64_t iW,
                         int64_t iC, int64_t oH, int64_t oW, int64_t oC,
                         int64_t kH, int64_t kW, int64_t strideH,
                         int64_t strideW, int64_t paddingH, int64_t paddingW,
                         int64_t dilationH = 1, int64_t dilationW = 1,
                         EpilogueEnum epilogue = EpilogueEnum::None);

inline const char *layout_enum_to_str(LayoutEnum a) {
  if (a == LayoutEnum::RowMajor) {
    return "row";
  } else if (a == LayoutEnum::ColumnMajor) {
    return "col";
  } else if (a == LayoutEnum::NHWC) {
    return "nhwc";
  } else if (a == LayoutEnum::NCHW) {
    return "nchw";
  }
  return "";
}

inline LayoutEnum str_to_layout_enum(const std::string &a) {
  if (a == "row") {
    return LayoutEnum::RowMajor;
  } else if (a == "col") {
    return LayoutEnum::ColumnMajor;
  } else if (a == "nhwc") {
    return LayoutEnum::NHWC;
  } else if (a == "nchw") {
    return LayoutEnum::NCHW;
  } else {
    return LayoutEnum::Invalid;
  }
}

inline const char *epilogue_enum_to_str(EpilogueEnum a) {
  if (a == EpilogueEnum::None) {
    return "";
  } else if (a == EpilogueEnum::Relu) {
    return "Relu";
  } else if (a == EpilogueEnum::Sigmoid) {
    return "Sigmoid";
  } else {
    return "Unknown";
  }
}
