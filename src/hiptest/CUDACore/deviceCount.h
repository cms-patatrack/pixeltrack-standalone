#ifndef HeterogenousCore_CUDAUtilities_deviceCount_h
#define HeterogenousCore_CUDAUtilities_deviceCount_h

#include "CUDACore/cudaCheck.h"

#include <hip/hip_runtime.h>

namespace cms {
  namespace hip {
    inline int deviceCount() {
      int ndevices;
      cudaCheck(hipGetDeviceCount(&ndevices));
      return ndevices;
    }
  }  // namespace hip
}  // namespace cms

#endif
