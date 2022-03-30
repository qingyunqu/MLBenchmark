#pragma once

#include <cassert>
#include <stdint.h>
#include <string>

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
  MatmulBias = 2,
  Conv2d = 4,
  Conv2dBias = 5,
};

enum class EpilogueEnum : uint32_t {
  None = 0,
  Relu = 1,
  Grelu = 2,
  Sigmoid = 3,
  Tanh = 4,
  Elu = 5,
  LeakyRelu = 6,
  HardSwish = 7,
};

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
  assert(false && "Invalid Layout");
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
  }
  return LayoutEnum::Invalid;
}

inline const char *epilogue_enum_to_str(EpilogueEnum a) {
  if (a == EpilogueEnum::None) {
    return "";
  } else if (a == EpilogueEnum::Relu) {
    return "Relu";
  } else if (a == EpilogueEnum::Grelu) {
    return "Grelu";
  } else if (a == EpilogueEnum::Sigmoid) {
    return "Sigmoid";
  } else if (a == EpilogueEnum::Tanh) {
    return "Tanh";
  } else if (a == EpilogueEnum::LeakyRelu) {
    return "LeakyRelu";
  } else if (a == EpilogueEnum::HardSwish) {
    return "HardSwish";
  }
  assert(false && "Unknown Epilogue");
}

inline const char *dtype_enum_to_str(DTypeEnum a) {
  if (a == DTypeEnum::Float32) {
    return "fp32";
  } else if (a == DTypeEnum::Float16) {
    return "fp16";
  } else if (a == DTypeEnum::BFloat16) {
    return "bf16";
  }
  assert(false && "Invalid DType");
}

inline const char *operation_enum_to_str(OperationEnum a) {
  if (a == OperationEnum::Matmul) {
    return "Matmul";
  } else if (a == OperationEnum::MatmulBias) {
    return "MatmulBias";
  } else if (a == OperationEnum::Conv2d) {
    return "Conv2d";
  } else if (a == OperationEnum::Conv2dBias) {
    return "Conv2dBias";
  }
  assert(false && "Invalid Operation");
}
