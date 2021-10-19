#ifndef HeterogeneousCore_AlpakaUtilities_EventCache_h
#define HeterogeneousCore_AlpakaUtilities_EventCache_h

#include <vector>

#include <cuda_runtime.h>

#include "AlpakaCore/ScopedSetDevice.h"
#include "AlpakaCore/SharedEventPtr.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/currentDevice.h"
#include "AlpakaCore/deviceCount.h"
#include "AlpakaCore/eventWorkHasCompleted.h"
#include "Framework/ReusableObjectHolder.h"

class CUDAService;

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  class EventCache {
  public:
    EventCache();

    // Gets a (cached) CUDA event for the current device. The event
    // will be returned to the cache by the shared_ptr destructor. The
    // returned event is guaranteed to be in the state where all
    // captured work has completed, i.e. cudaEventQuery() == cudaSuccess.
    //
    // This function is thread safe
    template <typename T_Acc>
    SharedEventPtr get(T_Acc acc) {
      const auto dev = currentDevice();
      auto event = makeOrGet(dev, acc);
      // captured work has completed, or a just-created event
      if (eventWorkHasCompleted(*(event.get()))) {
        return event;
      }

      // Got an event with incomplete captured work. Try again until we
      // get a completed (or a just-created) event. Need to keep all
      // incomplete events until a completed event is found in order to
      // avoid ping-pong with an incomplete event.
      std::vector<SharedEventPtr> ptrs{std::move(event)};
      bool completed;
      do {
        event = makeOrGet(dev, acc);
        completed = eventWorkHasCompleted(*(event.get()));
        if (not completed) {
          ptrs.emplace_back(std::move(event));
        }
      } while (not completed);
      return event;
    }

  private:
    friend class ::CUDAService;

    template <typename T_Acc>
    SharedEventPtr makeOrGet(int dev, T_Acc acc) {
      return cache_[dev].makeOrGet(
          [dev, acc]() { return std::make_unique<::ALPAKA_ACCELERATOR_NAMESPACE::Event>(acc); });
    }

    // not thread safe, intended to be called only from CUDAService destructor
    void clear();

    std::vector<edm::ReusableObjectHolder<::ALPAKA_ACCELERATOR_NAMESPACE::Event>> cache_;
  };

  // Gets the global instance of a EventCache
  // This function is thread safe
  EventCache& getEventCache();

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaUtilities_EventCache_h
