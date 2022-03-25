#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string>

#include "../check.h"
#include "util/util.h"

template <typename T, typename To, typename CompType = float>
bool CheckConv(T *input, T *filter, To *output, const std::string &layout,
               int64_t N, int64_t iC, int64_t iH, int64_t iW, int64_t oC,
               int64_t kH, int64_t kW, int64_t oH, int64_t oW, int64_t strideH,
               int64_t strideW, int64_t paddingH, int64_t paddingW,
               int64_t dilateH, int64_t dilateW, float eps) {
  T *h_input = (T *)malloc(N * iH * iW * iC * sizeof(T));
  T *h_filter = (T *)malloc(oC * kH * kW * iC * sizeof(T));
  To *h_output = (To *)malloc(N * oH * oW * oC * sizeof(To));
  CUDACHECK(cudaMemcpy(h_input, input, N * iH * iW * iC * sizeof(T),
                       cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_filter, filter, oC * kH * kW * iC * sizeof(T),
                       cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_output, output, N * oH * oW * oC * sizeof(To),
                       cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

  assert(dilateH == 1 && dilateW == 1);
  assert(layout == "NHWC" || layout == "NCHW");

  bool check = true;
  for (int64_t n = 0; n < N; n++) {
    for (int64_t oc = 0; oc < oC; oc++) {
      for (int64_t oh = 0; oh < oH; oh++) {
        for (int64_t ow = 0; ow < oW; ow++) {
          int64_t output_index = 0;
          if (layout == "NHWC") {
            output_index = n * oH * oW * oC + oh * oW * oC + ow * oC + oc;
          } else if (layout == "NCHW") {
            output_index = n * oC * oH * oW + oc * oH * oW + oh * oW + ow;
          }
          CompType result = static_cast<CompType>(0.f);
          for (int64_t ic = 0; ic < iC; ic++) {
            int64_t ih = oh * strideH - paddingH;
            int64_t iw = ow * strideW - paddingW;
            for (int64_t kh = 0; kh < kH; kh++) {
              for (int64_t kw = 0; kw < kW; kw++) {
                int64_t filter_index = 0;
                if (layout == "NHWC") {
                  filter_index =
                      oc * kH * kW * iC + kh * kW * iC + kw * iC + ic;
                } else if (layout == "NCHW") {
                  filter_index =
                      oc * iC * kH * kW + ic * kH * kW + kh * kW + kw;
                }
                int64_t t_ih = ih + kh;
                int64_t t_iw = iw + kw;
                if (t_ih >= 0 && t_iw >= 0 && t_ih < iH && t_iw < iW) {
                  int64_t input_index = 0;
                  if (layout == "NHWC") {
                    input_index =
                        n * iH * iW * iC + t_ih * iW * iC + t_iw * iC + ic;
                  } else if (layout == "NCHW") {
                    input_index =
                        n * iC * iH * iW + ic * iH * iW + t_ih * iW + t_iw;
                  }
                  result += static_cast<CompType>(h_input[input_index]) *
                            static_cast<CompType>(h_filter[filter_index]);
                }
              }
            }
          }
          check =
              EXPECT_NEAR(h_output[output_index], static_cast<To>(result), eps);
          if (!check) {
            goto EXIT;
          }
        }
      }
    }
  }

EXIT:
  free(h_input);
  free(h_filter);
  free(h_output);
  return check;
}
