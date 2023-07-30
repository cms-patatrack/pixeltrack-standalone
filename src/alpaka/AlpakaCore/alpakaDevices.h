#ifndef AlpakaCore_alpakaDevices_h
#define AlpakaCore_alpakaDevices_h

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/getDeviceIndex.h"

namespace cms::alpakatools {

  // alpaka host device
  inline const alpaka_common::PlatformHost platformHost{};
  inline const alpaka_common::DevHost host = alpaka::getDevByIdx(platformHost, 0u);

  // alpaka accelerator devices
  template <typename TPlatform>
  inline std::vector<alpaka::Dev<TPlatform>> devices;

  template <typename TPlatform>
  std::vector<alpaka::Dev<TPlatform>> enumerate() {
    assert(getDeviceIndex(host) == 0u);

    using Device = alpaka::Dev<TPlatform>;
    using Platform = TPlatform;
    platform = Platform{};

    std::vector<Device> devices;
    uint32_t n = alpaka::getDevCount(*platform);
    devices.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      devices.push_back(alpaka::getDevByIdx(*platform, i));
      assert(getDeviceIndex(devices.back()) == static_cast<int>(i));
    }
    return devices;
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_alpakaDevices_h
