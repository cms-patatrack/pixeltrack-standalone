#include <cassert>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"
#include "AlpakaCore/host.h"

namespace cms::alpakatools {

  // alpaka host platform and device

  // return the alpaka host platform
  alpaka_common::PlatformHost const& host_platform() {
    static const auto platform = alpaka_common::PlatformHost{};
    return platform;
  }

  // return the alpaka host device
  alpaka_common::DevHost const& host() {
    static const auto host = alpaka::getDevByIdx(host_platform(), 0u);
    // assert on the host index ?
    return host;
  }

}  // namespace cms::alpakatools
