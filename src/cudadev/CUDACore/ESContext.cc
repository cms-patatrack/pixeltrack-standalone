#include "CUDACore/ESContext.h"
#include "CUDACore/StreamCache.h"
#include "CUDACore/cudaCheck.h"

namespace cms::cuda {
  ESContext::ESContext(int device) : currentDevice_(device) {
    cudaCheck(cudaSetDevice(currentDevice_));
    stream_ = getStreamCache().get();
  }

}  // namespace cms::cuda
