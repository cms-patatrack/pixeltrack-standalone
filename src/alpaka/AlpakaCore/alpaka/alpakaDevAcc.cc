#include "AlpakaCore/alpakaDevAcc.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  static std::vector<ALPAKA_ACCELERATOR_NAMESPACE::Device> enumerate() {
    std::vector<ALPAKA_ACCELERATOR_NAMESPACE::Device> devices;
    uint32_t n = alpaka::getDevCount<ALPAKA_ACCELERATOR_NAMESPACE::Platform>();
    devices.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      devices.push_back(alpaka::getDevByIdx<Platform>(i));
    }
    return devices;
  }

  const std::vector<Device> devices = enumerate();
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
