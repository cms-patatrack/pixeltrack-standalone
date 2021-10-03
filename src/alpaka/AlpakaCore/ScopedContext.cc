#include "AlpakaCore/ScopedContext.h"

namespace {
  struct CallbackData {
    edm::WaitingTaskWithArenaHolder holder;
    int device;
  };
}  // namespace

namespace cms::alpakatools {
  namespace impl {
    void ScopedContextGetterBase::synchronizeStreams(int dataDevice,
                                                     Queue dataStream,
                                                     bool available,
                                                     alpaka::Event<cms::alpakatools::Queue> dataEvent) {
      if (dataDevice != device()) {
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
          alpaka::wait(stream(), dataEvent);
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(int device, Queue stream) {
      alpaka::enqueue(stream, [this, device]() {
        auto data = new CallbackData{waitingTaskHolder_, device};
        std::unique_ptr<CallbackData> guard{reinterpret_cast<CallbackData*>(data)};
        edm::WaitingTaskWithArenaHolder& waitingTaskHolder = guard->holder;
        int device2 = guard->device;
        waitingTaskHolder.doneWaiting(nullptr);
      });
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() {
    holderHelper_.enqueueCallback(device(), stream());
    if (contextState_) {
      contextState_->set(device(), streamPtr());
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
    //alpaka::enqueue(stream(), getEvent(ALPAKA_ACCELERATOR_NAMESPACE::Device).get());
    //TODO
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(device(), stream()); }
}  // namespace cms::alpakatools
