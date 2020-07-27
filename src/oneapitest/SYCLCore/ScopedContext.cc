#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SYCLCore/ScopedContext.h"
#include "chooseDevice.h"
#include <future>

namespace cms::sycl {
  namespace impl {
    ScopedContextBase::ScopedContextBase(edm::StreamID streamID) : currentDevice_(chooseDevice(streamID)) {
      stream_ = ::sycl::queue(currentDevice_);
    }

    ScopedContextBase::ScopedContextBase(const ProductBase& data) : currentDevice_(data.device()) {
      if (data.mayReuseStream()) {
        stream_ = data.stream();
      } else {
        stream_ = ::sycl::queue(currentDevice_);
      }
    }

    ScopedContextBase::ScopedContextBase(::sycl::device device, ::sycl::queue stream)
        : currentDevice_(device), stream_(stream) {
    }

    ////////////////////

    void ScopedContextGetterBase::synchronizeStreams(::sycl::device dataDevice,
                                                     ::sycl::queue dataStream,
                                                     bool available,
                                                     ::sycl::event dataEvent) {
      if (dataDevice != device()) {
        // Eventually replace with prefetch to current device (assuming unified memory works)
        // If we won't go to unified memory, need to figure out something else...
        throw std::runtime_error("Handling data from multiple devices is not yet supported");
      }

      if (dataStream != stream()) {
        // Different streams, need to synchronize
        if (not available) {
          // Event not yet occurred, so need to add synchronization
          // here. Sychronization is done by making the SYCL stream to
          // wait for an event, so all subsequent work in the stream
          // will run only after the event has "occurred" (i.e. data
          // product became available).
          dataEvent.wait();
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(::sycl::queue stream) {
      std::async([&]() {
          stream.wait();
          waitingTaskHolder_.doneWaiting(nullptr);
      });
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() {
    holderHelper_.enqueueCallback(stream());
    if (contextState_) {
      contextState_->set(device(), stream());
    }
  }

  void ScopedContextAcquire::throwNoState() {
    throw std::runtime_error(
        "Calling ScopedContextAcquire::insertNextTask() requires ScopedContextAcquire to be constructed with "
        "ContextState, but that was not the case");
  }

  ////////////////////

  ScopedContextProduce::~ScopedContextProduce() {
    // Intentionally not checking the return value to avoid throwing
    // exceptions. If this call would fail, we should get failures
    // elsewhere as well.
    /*
    DPCT1012:15: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
    */
    auto event__get_ct1 = clock();
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(stream()); }
}  // namespace cms::sycl
