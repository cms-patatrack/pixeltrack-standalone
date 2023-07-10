#ifndef SYCLCore_getPlatformIndex
#define SYCLCore_getPlatformIndex

#include <sycl/sycl.hpp>

#include "SYCLCore/chooseDevice.h"

namespace cms::sycltools {
  inline int getPlatformIndex(sycl::platform const& platform) {
    auto const& platforms = enumeratePlatforms();
    auto position = std::find(platforms.begin(), platforms.end(), platform);
    int index = position - platforms.begin();
    return index;
  }
}  // namespace cms::sycltools

#endif  // SYCLCore_getPlatformIndex