#include "CUDACore/Context.h"

#include "CUDACore/StreamCache.h"
#include "CUDACore/cudaCheck.h"

#include "chooseDevice.h"

namespace {
  struct CallbackData {
    edm::WaitingTaskWithArenaHolder holder;
    int device;
  };

  void CUDART_CB cudaContextCallback(cudaStream_t streamId, cudaError_t status, void* data) {
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
    Context::Context(edm::StreamID streamID) : Context(chooseDevice(streamID)) {}

    Context::Context(int device) : currentDevice_(device) { cudaCheck(cudaSetDevice(currentDevice_)); }

    Context::Context(int device, SharedStreamPtr stream)
        : currentDevice_(device), stream_(std::make_shared<impl::StreamSharingHelper>(std::move(stream))) {
      cudaCheck(cudaSetDevice(currentDevice_));
    }

    void Context::initialize() { stream_ = std::make_shared<impl::StreamSharingHelper>(getStreamCache().get()); }

    void Context::initialize(const ProductBase& data) {
      SharedStreamPtr stream;
      if (data.mayReuseStream()) {
        stream = data.streamPtr();
      } else {
        stream = getStreamCache().get();
      }
      stream_ = std::make_shared<impl::StreamSharingHelper>(std::move(stream));
    }

    ////////////////////

    void ContextGetterBase::synchronizeStreams(int dataDevice,
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

    ////////////////////

    void ContextHolderHelper::enqueueCallback(cudaStream_t stream) {
      cudaCheck(cudaStreamAddCallback(stream, cudaContextCallback, new CallbackData{waitingTaskHolder_, device_}, 0));
    }
  }  // namespace impl

  ////////////////////

  void AcquireContext::commit() { holderHelper_.enqueueCallback(stream()); }

  ////////////////////

  void ProduceContext::commit() { cudaCheck(cudaEventRecord(event_.get(), stream())); }

  ////////////////////

  void TaskContext::commit() { holderHelper_.enqueueCallback(stream()); }
}  // namespace cms::cuda
