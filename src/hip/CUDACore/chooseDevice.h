#ifndef HeterogeneousCore_CUDACore_chooseDevice_h
#define HeterogeneousCore_CUDACore_chooseDevice_h

#include "Framework/Event.h"

namespace cms::hip {
  int chooseDevice(edm::StreamID id);
}

#endif
