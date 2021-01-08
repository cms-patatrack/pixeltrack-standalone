#ifndef HeterogenousCore_CUDAUtilities_currentDevice_h
#define HeterogenousCore_CUDAUtilities_currentDevice_h

#include "CUDACore/cudaCheck.h"

#include <hip/hip_runtime.h>

namespace cms {
  namespace hip {
    inline int currentDevice() {
      int dev;
      cudaCheck(hipGetDevice(&dev));
      return dev;
    }
  }  // namespace hip
}  // namespace cms

#endif
