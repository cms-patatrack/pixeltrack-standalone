#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"
#include "AlpakaCore/alpaka/devices.h"

namespace cms::alpakatools {

  // alpaka accelerator platform and devices
  // Note: even when TPlatform is the same as PlatformHost these functions return different objects
  // than host_platform() and host()

  // return the alpaka accelerator platform for the given platform
  template <typename TPlatform, typename>
  TPlatform const& platform() {
    static const auto platform = TPlatform{};
    return platform;
  }

  // return the alpaka accelerator devices for the given platform
  template <typename TPlatform, typename>
  std::vector<alpaka::Dev<TPlatform>> const& devices() {
    static const auto devices = alpaka::getDevs(platform<TPlatform>());
    // assert on device index ?
    return devices;
  }

  // explicit template instantiation definitions
  template
  ALPAKA_ACCELERATOR_NAMESPACE::Platform const& platform<ALPAKA_ACCELERATOR_NAMESPACE::Platform, void>();

  template
  std::vector<ALPAKA_ACCELERATOR_NAMESPACE::Device> const& devices<ALPAKA_ACCELERATOR_NAMESPACE::Platform, void>();

}  // namespace cms::alpakatools
