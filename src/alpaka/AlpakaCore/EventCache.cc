
#include "AlpakaCore/EventCache.h"
#include "AlpakaCore/ScopedSetDevice.h"

namespace cms::alpakatools {
  // EventCache should be constructed by the first call to
  // getEventCache() only if we have CUDA devices present
  EventCache::EventCache() : cache_(cms::alpakatools::deviceCount()) {}

  void EventCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // EventCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(cms::alpakatools::deviceCount());
  }

  EventCache& getEventCache() {
    // the public interface is thread safe
    static EventCache cache;
    return cache;
  }
}  // namespace cms::alpakatools
