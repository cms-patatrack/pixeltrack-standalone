#ifndef AlpakaCore_host_h
#define AlpakaCore_host_h

#include "AlpakaCore/common.h"

namespace cms::alpakatools {

  // alpaka host platform and device

  // return the alpaka host platform
  alpaka_common::PlatformHost const& host_platform();

  // return the alpaka host device
  alpaka_common::DevHost const& host();

}  // namespace cms::alpakatools

#endif  // AlpakaCore_host_h
