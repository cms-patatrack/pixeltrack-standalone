#ifndef ALPAKADEVICEACC_H
#define ALPAKADEVICEACC_H

#include <vector>

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  template <typename TPlatform>
  std::vector<alpaka::Dev<TPlatform>> devices;

  template <typename TPlatform>
  std::vector<alpaka::Dev<TPlatform>> enumerate() {
    using Device = alpaka::Dev<TPlatform>;
    using Platform = TPlatform;

    std::vector<Device> devices;
    uint32_t n = alpaka::getDevCount<Platform>();
    devices.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      devices.push_back(alpaka::getDevByIdx<Platform>(i));
    }
    return devices;
  }

}  // namespace cms::alpakatools

#endif  // ALPAKADEVICEACC_H
