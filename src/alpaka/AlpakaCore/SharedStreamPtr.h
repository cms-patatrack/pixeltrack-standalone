#ifndef HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h
#define HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h

#include <memory>
#include <type_traits>
#include "AlpakaCore/alpakaConfigCommon.h"

namespace cms {
  namespace alpakatools {
    // cudaStream_t itself is a typedef for a pointer, for the use with
    // edm::ReusableObjectHolder the pointed-to type is more interesting
    // to avoid extra layer of indirection
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Queue = alpaka::QueueCudaRtNonBlocking;
#elif defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    using Queue = alpaka::QueueCpuNonBlocking;
#else
  using Queue = alpaka::QueueCpuBlocking;
#endif
    using SharedStreamPtr = std::shared_ptr<std::remove_pointer_t<Queue>>;
  }  // namespace alpakatools
}  // namespace cms

#endif