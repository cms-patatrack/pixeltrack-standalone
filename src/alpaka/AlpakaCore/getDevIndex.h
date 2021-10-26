#ifndef HeterogeneousCore_AlpakaUtilities_getDevIndex_h
#define HeterogeneousCore_AlpakaUtilities_getDevIndex_h

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  // generic interface, for DevOacc and DevOmp5
  template <typename Device>
  inline int getDevIndex(Device const& device) {
    return device.iDevice();
  }

  // overload for DevCpu
  inline int getDevIndex(alpaka::DevCpu const& device) { return 0; }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  // overload for DevCudaRt
  inline int getDevIndex(alpaka::DevCudaRt const& device) { return device.m_iDevice; }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaUtilities_getDevIndex_h
