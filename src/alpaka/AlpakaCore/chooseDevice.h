#ifndef HeterogeneousCore_AlpakaCore_chooseDevice_h
#define HeterogeneousCore_AlpakaCore_chooseDevice_h

#include "AlpakaCore/config.h"
#include "AlpakaCore/alpaka/devices.h"
#include "Framework/Event.h"

namespace cms::alpakatools {

  template <typename TPlatform>
  alpaka::Dev<TPlatform> const& chooseDevice(edm::StreamID id) {
    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of devices
    // (and even then there is only implicit load balancing).

    // TODO: improve the "assignment" logic
    auto const& devices = cms::alpakatools::devices<TPlatform>();
    return devices[id % devices.size()];
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_chooseDevice_h
