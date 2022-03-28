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
void profile_gemm_gemm(Manifest &manifest, int64_t m0, int64_t n0, int64_t k0,
                       int64_t n1, LayoutEnum layout_a, LayoutEnum layout_b,
                       LayoutEnum layout_c);

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

template <typename TA, typename TB, typename TC, typename CompOn>
void profile_conv2d_conv2d(Manifest &manifest, int64_t N0, int64_t iH0,
                           int64_t iW0, int64_t iC0, int64_t oH0, int64_t oW0,
                           int64_t oC0, int64_t kH0, int64_t kW0,
                           int64_t strideH0, int64_t strideW0,
                           int64_t paddingH0, int64_t paddingW0,
                           int64_t dilationH0, int64_t dilationW0, int64_t N1,
                           int64_t iH1, int64_t iW1, int64_t iC1, int64_t oH1,
                           int64_t oW1, int64_t oC1, int64_t kH1, int64_t kW1,
                           int64_t strideH1, int64_t strideW1,
                           int64_t paddingH1, int64_t paddingW1,
                           int64_t dilationH1, int64_t dilationW1);
