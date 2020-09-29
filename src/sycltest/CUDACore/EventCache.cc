#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUDACore/EventCache.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/eventWorkHasCompleted.h"
#include "CUDACore/ScopedSetDevice.h"

namespace cms::cuda {
  void EventCache::Deleter::operator()(sycl::event event) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
      /*
      DPCT1027:77: The call to cudaEventDestroy was replaced with 0, because this call is redundant in DPC++.
      */
      cudaCheck(0);
    }
  }

  // EventCache should be constructed by the first call to
  // getEventCache() only if we have CUDA devices present
  EventCache::EventCache() : cache_(deviceCount()) {}

  SharedEventPtr EventCache::get() {
    const auto dev = currentDevice();
    auto event = makeOrGet(dev);
    // captured work has completed, or a just-created event
    if (eventWorkHasCompleted(event.get())) {
      return event;
    }

    // Got an event with incomplete captured work. Try again until we
    // get a completed (or a just-created) event. Need to keep all
    // incomplete events until a completed event is found in order to
    // avoid ping-pong with an incomplete event.
    std::vector<SharedEventPtr> ptrs{std::move(event)};
    bool completed;
    do {
      event = makeOrGet(dev);
      completed = eventWorkHasCompleted(event.get());
      if (not completed) {
        ptrs.emplace_back(std::move(event));
      }
    } while (not completed);
    return event;
  }

  SharedEventPtr EventCache::makeOrGet(int dev) {
    return cache_[dev].makeOrGet([dev]() {
      try {
        sycl::event event;
      // it should be a bit faster to ignore timings
        /*
      DPCT1027:78: The call to cudaEventCreateWithFlags was replaced with 0, because this call is redundant in DPC++.
      */
        cudaCheck(0);
      return std::unique_ptr<BareEvent, Deleter>(event, Deleter{dev});
      }
      catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
      }
    });
  }

  void EventCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // EventCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(deviceCount());
  }

  EventCache& getEventCache() {
    // the public interface is thread safe
    static EventCache cache;
    return cache;
  }
}  // namespace cms::cuda
