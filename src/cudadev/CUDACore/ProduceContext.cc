#include "CUDACore/ProduceContext.h"
#include "CUDACore/cudaCheck.h"

namespace cms::cuda {
  void ProduceContext::commit() { cudaCheck(cudaEventRecord(event_.get(), stream())); }
}
