#include <CL/sycl.hpp>

#include "chooseDevice.h"
#include "deviceCount.h"

namespace cms::sycl {
  ::sycl::device chooseDevice(edm::StreamID id) {
    // For the moment, just return the default device.
    // TODO: improve the "assignment" logic
    static ::sycl::default_selector selector;
    static ::sycl::device device{selector};
    
    return device;
  }
}  // namespace cms::sycl
