#ifndef HeterogenousCore_AlpakaUtilities_currentDevice_h
#define HeterogenousCore_AlpakaUtilities_currentDevice_h

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  inline int currentDevice() {
    int dev = 0;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    cudaGetDevice(&dev);
#endif
    return dev;
  }

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif
