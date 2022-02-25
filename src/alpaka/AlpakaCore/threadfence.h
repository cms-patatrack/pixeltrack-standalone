#ifndef AlpakaCore_threadfence_h
#define AlpakaCore_threadfence_h

#include <atomic>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools {

  // device-wide memory fence
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(TAcc const& acc) {
    alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_threadfence_h
