#pragma once

#include <stdint.h>

enum class DTypeEnum : uint32_t {
  Invalid = 0,
  Float32 = 1,
  Int32 = 2,
  Int64 = 3,
  UInt8 = 4,
  UInt32 = 5,
  Float16 = 6,
  BFloat16 = 7,
  Float64 = 8
};

enum class LayoutEnum : uint32_t {
  Invalid = 0,
  RowMajor = 1,
  ColumnMajor = 2,
  NCHW = 3,
  NHWC = 4
};

enum class OperationEnum : uint32_t {
  Invalid = 0,
  Matmul = 1,
  MatmulBiasRelu = 2,
  Conv2d = 3,
  Conv2dBiasRelu = 4
};
