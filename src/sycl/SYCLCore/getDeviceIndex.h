#ifndef SYCLCore_getDeviceIndex_h
#define SYCLCore_getDeviceIndex_h

#include <CL/sycl.hpp>

#include "SYCLCore/chooseDevice.h"

namespace cms::sycltools {
  inline int getDeviceIndex(sycl::device const& device) {
    auto const& devices = enumerateDevices();
    auto position = std::find(devices.begin(), devices.end(), device);
    int index = position - devices.begin();
    return index;
  }
}  // namespace cms::sycltools

#endif  // SYCLCore_getDeviceIndex_h