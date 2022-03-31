#pragma once

#include <string>
#include <cassert>

//===----------------------------------------------------------------------===//
// General Op
//===----------------------------------------------------------------------===//

template <typename T, typename To = T> class Op {
public:
  virtual bool Check() = 0;
  virtual void AllocWorkspace() { assert(false && "should not reach this"); }
  virtual void SetArgument(T *) { assert(false && "should not reach this"); }
  virtual void SetArgument(T *, T *, To *) {
    assert(false && "should not reach this");
  }
  virtual void SetArgument(T *, T *, To *, To *) {
    assert(false && "should not reach this");
  }
  virtual void Run() { assert(false && "should not reach this"); }
};
