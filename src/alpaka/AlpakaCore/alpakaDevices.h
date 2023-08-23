#ifndef AlpakaCore_alpakaDevices_h
#define AlpakaCore_alpakaDevices_h

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"
#include "AlpakaCore/getDeviceIndex.h"

namespace cms::alpakatools {

  // alpaka host platform and device
  inline const alpaka_common::PlatformHost platformHost{};
  inline const alpaka_common::DevHost host = alpaka::getDevByIdx(platformHost, 0u);

  // alpaka accelerator platform and devices
  // these objects are filled by a call to cms::alpakatools::initialise<TPlatform>()
  template <typename TPlatform>
  inline std::optional<TPlatform> platform;

  template <typename TPlatform>
  inline std::vector<alpaka::Dev<TPlatform>> devices;

}  // namespace cms::alpakatools

#endif  // AlpakaCore_alpakaDevices_h
