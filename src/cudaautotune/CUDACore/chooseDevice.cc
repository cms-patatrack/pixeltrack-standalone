#include "chooseDevice.h"
#include "deviceCount.h"

namespace cms::cuda {
  int chooseDevice(edm::StreamID id) {
    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve the "assignment" logic
    return id % deviceCount();
  }
}  // namespace cms::cuda
