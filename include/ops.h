#pragma once

#include <string>
#include <cassert>

//===----------------------------------------------------------------------===//
// General Op
//===----------------------------------------------------------------------===//

template <typename T, typename To = T> class Op {
public:
  virtual bool Check() = 0;
  virtual void Initialize() {
    assert(false && "should not reach this");
  }
  virtual void Run(T *, T *, To *) {
    assert(false && "should not reach this");
  }
  virtual void Run(T * ,T *, To *, To *) {
    assert(false && "should not reach this");
  }
};
