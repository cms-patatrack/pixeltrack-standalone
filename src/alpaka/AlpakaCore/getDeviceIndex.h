#ifndef AlpakaCore_getDeviceIndex_h
#define AlpakaCore_getDeviceIndex_h

#include <optional>

#include <alpaka/alpaka.hpp>
#include "AlpakaCore/config.h"

namespace cms::alpakatools {

  // generic interface, for DevOacc and DevOmp5
  template <typename Device>
  inline int getDeviceIndex(Device const& device) {
    return device.iDevice();
  }

  // overload for DevCpu
  inline int getDeviceIndex(alpaka::DevCpu const& device) { return 0; }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  // overload for DevCudaRt
  inline int getDeviceIndex(alpaka::DevCudaRt const& device) { return alpaka::getNativeHandle(device); }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  // overload for DevHipRt
  inline int getDeviceIndex(alpaka::DevHipRt const& device) { return alpaka::getNativeHandle(device); }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_SYCL_ONEAPI_CPU
  // overload for DevGenericSycl
  inline int getDeviceIndex(alpaka::DevCpuSycl const& device) {
    return std::find(platform<alpaka::PlatformCpuSycl>().syclDevices().begin(),
                     platform<alpaka::PlatformCpuSycl>().syclDevices().end(),
                     device.getNativeHandle().first) -
           platform<alpaka::PlatformCpuSycl>().syclDevices().begin();
  }
#endif // ALPAKA_SYCL_ONEAPI_CPU

#ifdef ALPAKA_SYCL_ONEAPI_GPU
  inline int getDeviceIndex(alpaka::DevGpuSyclIntel const& device) {
    return std::find(platform<alpaka::PlatformGpuSyclIntel>().syclDevices().begin(),
                     platform<alpaka::PlatformGpuSyclIntel>().syclDevices().end(),
                     device.getNativeHandle().first) -
           platform<alpaka::PlatformGpuSyclIntel>().syclDevices().begin();
  }
#endif  // ALPAKA_SYCL_ONEAPI_GPU

}  // namespace cms::alpakatools

#endif  // AlpakaCore_getDeviceIndex_h
