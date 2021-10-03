
#ifndef HeterogenousCore_AlpakaUtilities_currentDevice_h
#define HeterogenousCore_AlpakaUtilities_currentDevice_h

#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {
    inline int currentDevice() {
      int dev = 0;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      cudaGetDevice(&dev);
#endif
      return dev;
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
