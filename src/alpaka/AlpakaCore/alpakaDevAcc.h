#ifndef ALPAKADEVICEACC_H
#define ALPAKADEVICEACC_H

#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfigAcc.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  extern std::vector<Device> devices;
}

template <typename TDevice>
std::vector<TDevice> enumerate() {
  using Device = TDevice;
  using Platform = alpaka::Pltf<Device>;

  std::vector<Device> devices;
  uint32_t n = alpaka::getDevCount<Platform>();
  devices.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    devices.push_back(alpaka::getDevByIdx<Platform>(i));
  }
  return devices;
}

#endif  // ALPAKADEVICEACC_H
