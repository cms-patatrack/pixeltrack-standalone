#include <vector>

#include <CL/sycl.hpp>

#include "chooseDevice.h"

namespace cms::cuda {
  sycl::device chooseDevice(edm::StreamID id) {
    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve the "assignment" logic
    static std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::all);

    return devices[id % devices.size()];
  }
}  // namespace cms::cuda
