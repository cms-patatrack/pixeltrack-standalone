#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/ScopedContext.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  namespace impl {
    void ScopedContextGetterBase::synchronizeStreams(Queue& dataStream,
                                                     bool available,
                                                     alpaka::Event<Queue> dataEvent) {
      if (dataStream != stream()) {
        // Different streams, check if the underlying device is the same
        if (alpaka::getDev(dataStream) != device()) {
          // Eventually replace with prefetch to current device (assuming unified memory works)
          // If we won't go to unified memory, need to figure out something else...
          throw std::runtime_error("Handling data from multiple devices is not yet supported");
        }

        // Synchronize the two streams
        if (not available) {
          // Event not yet occurred, so need to add synchronization
          // here. Sychronization is done by making the CUDA stream to
          // wait for an event, so all subsequent work in the stream
          // will run only after the event has "occurred" (i.e. data
          // product became available).
          alpaka::wait(stream(), dataEvent);
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(ScopedContextBase::Queue& stream) {
      alpaka::enqueue(stream, [holder = waitingTaskHolder_]() {
        // TODO: The functor is required to be const, so can't use
        // 'mutable', so I'm copying the object as a workaround. I
        // wonder if there are any wider implications.
        auto h = holder;
        h.doneWaiting(nullptr);
      });
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() {
    holderHelper_.enqueueCallback(stream());
    if (contextState_) {
      contextState_->set(streamPtr());
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
    //cudaEventRecord(event_.get(), stream());
    //alpaka::enqueue(stream(), getEvent(::ALPAKA_ACCELERATOR_NAMESPACE::Device).get());
    //TODO
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(stream()); }

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE
