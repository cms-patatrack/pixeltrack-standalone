#include <chrono>
#include <future>

#include <CL/sycl.hpp>

#include "SYCLCore/ScopedContext.h"
#include "chooseDevice.h"

namespace cms::sycltools {
  namespace impl {

    void sycl_exception_handler(cl::sycl::exception_list exceptions) {
      std::ostringstream msg;
      msg << "Caught asynchronous SYCL exception:";
      for (auto const &exc_ptr : exceptions) {
        try {
          std::rethrow_exception(exc_ptr);
        } catch (cl::sycl::exception const &e) {
          msg << '\n' << e.what();
        }
        throw std::runtime_error(msg.str());
      }
    }

    ScopedContextBase::ScopedContextBase(edm::StreamID streamID)
        : stream_(chooseDevice(streamID), sycl_exception_handler, sycl::property::queue::in_order()) {}

    ScopedContextBase::ScopedContextBase(ProductBase const &data)
        : stream_(data.mayReuseStream()
                      ? data.stream()
                      : sycl::queue{data.device(), sycl_exception_handler, sycl::property::queue::in_order()}) {}

    ScopedContextBase::ScopedContextBase(sycl::queue stream) : stream_(std::move(stream)) {}

    ////////////////////

    void ScopedContextGetterBase::synchronizeStreams(sycl::queue dataStream, bool available, sycl::event dataEvent) {
      if (dataStream.get_device() != device()) {
        // Eventually replace with prefetch to current device (assuming unified memory works)
        // If we won't go to unified memory, need to figure out something else...
        throw std::runtime_error("Handling data from multiple devices is not yet supported");
      }

      if (dataStream != stream()) {
        // Different streams, need to synchronize
        if (not available) {
          // Event not yet occurred, so need to add synchronization
          // here. Sychronization is done by making the CUDA stream to
          // wait for an event, so all subsequent work in the stream
          // will run only after the event has "occurred" (i.e. data
          // product became available).
          dataEvent.wait();
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(sycl::queue stream) {
      // TODO: make truly asynchronous
      try {
        stream.wait_and_throw();
        waitingTaskHolder_.doneWaiting(nullptr);
      } catch(...) {
        waitingTaskHolder_.doneWaiting(std::current_exception());
      }
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() {
    holderHelper_.enqueueCallback(stream());
    if (contextState_) {
      contextState_->set(stream());
    }
  }

  void ScopedContextAcquire::throwNoState() {
    throw std::runtime_error(
        "Calling ScopedContextAcquire::insertNextTask() requires ScopedContextAcquire to be constructed with "
        "ContextState, but that was not the case");
  }

  ////////////////////

  ScopedContextProduce::~ScopedContextProduce() {
    // the barrier should be a no-op on an ordered queue, but is used to mark the end of the data processing
    event_ = stream().submit_barrier();
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(stream()); }
}  // namespace cms::sycltools
