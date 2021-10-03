#include "deviceCount.h"

namespace cms::alpakatools {
  int deviceCount() {
    int ndevices = 1;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    ndevices = alpaka::getDevCount<ALPAKA_ACCELERATOR_NAMESPACE::Platform>();
#endif
    return ndevices;
  }
}  // namespace cms::alpakatools
