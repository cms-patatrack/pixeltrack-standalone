#ifndef HeterogenousCore_CUDAUtilities_currentDevice_h
#define HeterogenousCore_CUDAUtilities_currentDevice_h

#include "CUDACore/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    inline int currentDevice() {
      int dev;
      cudaCheck(cudaGetDevice(&dev));
      return dev;
    }
  }  // namespace cuda
}  // namespace cms

#endif
