#ifndef AlpakaCore_devices_h
#define AlpakaCore_devices_h

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"

namespace cms::alpakatools {

  // alpaka host platform and device

  // return the alpaka host platform
  inline alpaka_common::PlatformHost const& host_platform() {
    static const auto platform = alpaka_common::PlatformHost{};
    return platform;
  }

  // return the alpaka host device
  inline alpaka_common::DevHost const& host() {
    static const auto host = alpaka::getDevByIdx(host_platform(), 0u);
    return host;
  }

  // alpaka accelerator platform and devices
  // Note: even when TPlatform is the same as PlatformHost these functions return different objects
  // than host_platform() and host()

  // return the alpaka accelerator platform for the given platform
  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  inline TPlatform const& platform() {
    static const auto platform = TPlatform{};
    return platform;
  }

  // return the alpaka accelerator devices for the given platform
  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  inline std::vector<alpaka::Dev<TPlatform>> const& devices() {
    static const auto devices = alpaka::getDevs(platform<TPlatform>());
    return devices;
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_devices_h
