#ifndef HeterogeneousCore_AlpakaUtilities_EventCache_h
#define HeterogeneousCore_AlpakaUtilities_EventCache_h

#include <memory>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>
#include "alpaka/alpaka_ex.hpp"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/getDevIndex.h"
#include "Framework/ReusableObjectHolder.h"

namespace cms::alpakatools {

  template <typename Event>
  class EventCache {
  public:
    using Device = alpaka::Dev<Event>;
    using Platform = alpaka::Pltf<Device>;

    // EventCache should be constructed by the first call to
    // getEventCache() only if we have CUDA devices present
    EventCache() : cache_(alpaka::getDevCount<Platform>()) {}

    // Gets a (cached) CUDA event for the current device. The event
    // will be returned to the cache by the shared_ptr destructor. The
    // returned event is guaranteed to be in the state where all
    // captured work has completed, i.e. cudaEventQuery() == cudaSuccess.
    //
    // This function is thread safe
    template <typename Device>
    std::shared_ptr<Event> get(Device dev) {
      auto event = makeOrGet(dev);
      // captured work has completed, or a just-created event
      if (alpaka::isComplete(*event)) {
        return event;
      }

      // Got an event with incomplete captured work. Try again until we
      // get a completed (or a just-created) event. Need to keep all
      // incomplete events until a completed event is found in order to
      // avoid ping-pong with an incomplete event.
      std::vector<std::shared_ptr<Event>> ptrs{std::move(event)};
      bool completed;
      do {
        event = makeOrGet(dev);
        completed = alpaka::isComplete(*event);
        if (not completed) {
          ptrs.emplace_back(std::move(event));
        }
      } while (not completed);
      return event;
    }

  private:
    std::shared_ptr<Event> makeOrGet(Device dev) {
      return cache_[::cms::alpakatools::getDevIndex(dev)].makeOrGet([dev]() { return std::make_unique<Event>(dev); });
    }

    // not thread safe, intended to be called only from CUDAService destructor
    void clear() {
      // Reset the contents of the caches, but leave an
      // edm::ReusableObjectHolder alive for each device. This is needed
      // mostly for the unit tests, where the function-static
      // EventCache lives through multiple tests (and go through
      // multiple shutdowns of the framework).
      cache_.clear();
      cache_.resize(alpaka::getDevCount<Platform>());
    }

    std::vector<edm::ReusableObjectHolder<Event>> cache_;
  };

  // Gets the global instance of a EventCache
  // This function is thread safe
  template <typename Event>
  EventCache<Event>& getEventCache() {
    // the public interface is thread safe
    static EventCache<Event> cache;
    return cache;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaUtilities_EventCache_h
