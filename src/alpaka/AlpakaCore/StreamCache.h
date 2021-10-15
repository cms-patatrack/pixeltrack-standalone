#ifndef HeterogeneousCore_AlpakaUtilities_StreamCache_h
#define HeterogeneousCore_AlpakaUtilities_StreamCache_h

#include <vector>

#include <cuda_runtime.h>

#include "AlpakaCore/ScopedSetDevice.h"
#include "AlpakaCore/SharedStreamPtr.h"
#include "AlpakaCore/currentDevice.h"
#include "AlpakaCore/deviceCount.h"
#include "Framework/ReusableObjectHolder.h"

class CUDAService;

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  class StreamCache {
  public:
    StreamCache();

    // Gets a (cached) CUDA stream for the current device. The stream
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    template <typename T_Acc>
    ALPAKA_FN_HOST SharedStreamPtr get(T_Acc acc) {
      const auto dev = currentDevice();
      return cache_[dev].makeOrGet(
          [dev, acc]() { return std::make_unique<::ALPAKA_ACCELERATOR_NAMESPACE::Queue>(acc); });
    }

  private:
    friend class ::CUDAService;
    // not thread safe, intended to be called only from CUDAService destructor
    void clear();

    std::vector<edm::ReusableObjectHolder<::ALPAKA_ACCELERATOR_NAMESPACE::Queue>> cache_;
  };

  // Gets the global instance of a StreamCache
  // This function is thread safe
  StreamCache& getStreamCache();

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaUtilities_StreamCache_h
