#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/deviceCount.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  int deviceCount() {
    int ndevices = 1;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    ndevices = alpaka::getDevCount<::ALPAKA_ACCELERATOR_NAMESPACE::Platform>();
#endif
    return ndevices;
  }

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE
