#ifndef HeterogeneousCore_SYCLCore_chooseDevice_h
#define HeterogeneousCore_SYCLCore_chooseDevice_h

#include <CL/sycl.hpp>

#include "Framework/Event.h"

namespace cms::sycl {
  ::sycl::device chooseDevice(edm::StreamID id);
}

#endif
