#include "CUDACore/ScopedContext.h"

#include "CUDACore/StreamCache.h"
#include "CUDACore/cudaCheck.h"

#include "chooseDevice.h"

namespace {
  struct CallbackData {
    edm::WaitingTaskWithArenaHolder holder;
    int device;
  };

  void CUDART_CB cudaScopedContextCallback(cudaStream_t streamId, cudaError_t status, void* data) {
    std::unique_ptr<CallbackData> guard{reinterpret_cast<CallbackData*>(data)};
    edm::WaitingTaskWithArenaHolder& waitingTaskHolder = guard->holder;
    int device = guard->device;
    if (status == cudaSuccess) {
      //std::cout << " GPU kernel finished (in callback) device " << device << " CUDA stream "
      //          << streamId << std::endl;
      waitingTaskHolder.doneWaiting(nullptr);
    } else {
      // wrap the exception in a try-catch block to let GDB "catch throw" break on it
      try {
        auto error = cudaGetErrorName(status);
        auto message = cudaGetErrorString(status);
        throw std::runtime_error("Callback of CUDA stream " +
                                 std::to_string(reinterpret_cast<unsigned long>(streamId)) + " in device " +
                                 std::to_string(device) + " error " + std::string(error) + ": " + std::string(message));
      } catch (std::exception&) {
        waitingTaskHolder.doneWaiting(std::current_exception());
      }
    }
  }
}  // namespace

namespace cms::cuda {
  namespace impl {
    ScopedContextBase::ScopedContextBase(edm::StreamID streamID) : currentDevice_(chooseDevice(streamID)) {
      cudaCheck(cudaSetDevice(currentDevice_));
      stream_ = getStreamCache().get();
    }

    ScopedContextBase::ScopedContextBase(const ProductBase& data) : currentDevice_(data.device()) {
      cudaCheck(cudaSetDevice(currentDevice_));
      if (data.mayReuseStream()) {
        stream_ = data.streamPtr();
      } else {
        stream_ = getStreamCache().get();
      }
    }

    ScopedContextBase::ScopedContextBase(int device, SharedStreamPtr stream)
        : currentDevice_(device), stream_(std::move(stream)) {
      cudaCheck(cudaSetDevice(currentDevice_));
    }

    ////////////////////

    void ScopedContextGetterBase::synchronizeStreams(int dataDevice,
                                                     cudaStream_t dataStream,
                                                     bool available,
                                                     cudaEvent_t dataEvent) {
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
          cudaCheck(cudaStreamWaitEvent(stream(), dataEvent, 0), "Failed to make a stream to wait for an event");
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(int device, cudaStream_t stream) {
      cudaCheck(
          cudaStreamAddCallback(stream, cudaScopedContextCallback, new CallbackData{waitingTaskHolder_, device}, 0));
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
    cudaEventRecord(event_.get(), stream());
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(device(), stream()); }
}  // namespace cms::cuda
