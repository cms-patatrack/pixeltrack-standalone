#include "CUDACore/AcquireContext.h"

namespace cms::cuda {
  void AcquireContext::commit() { holderHelper_.enqueueCallback(stream()); }
}
