#ifndef HeterogeneousCore_CUDACore_chooseDevice_h
#define HeterogeneousCore_CUDACore_chooseDevice_h

#include "Framework/Event.h"

namespace cms::cuda {
  int chooseDevice(edm::StreamID id);
}

#endif
