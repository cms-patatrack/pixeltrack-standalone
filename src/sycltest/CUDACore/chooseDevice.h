#ifndef HeterogeneousCore_CUDACore_chooseDevice_h
#define HeterogeneousCore_CUDACore_chooseDevice_h

#include <CL/sycl.hpp>

#include "Framework/Event.h"

namespace cms::cuda {
  std::vector<sycl::device> const& enumerateDevices(bool verbose = false);
  sycl::device chooseDevice(edm::StreamID id);
}

#endif
