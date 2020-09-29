#include <chrono>
#include <future>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "CUDACore/ScopedContext.h"
#include "chooseDevice.h"


namespace {
  struct CallbackData {
    edm::WaitingTaskWithArenaHolder holder;
    int device;
  };

  void CUDART_CB cudaScopedContextCallback(sycl::queue* streamId, int status, void* data) {
    std::unique_ptr<CallbackData> guard{reinterpret_cast<CallbackData*>(data)};
    edm::WaitingTaskWithArenaHolder& waitingTaskHolder = guard->holder;
    int device = guard->device;
    if (status == 0) {
      //std::cout << " GPU kernel finished (in callback) device " << device << " CUDA stream "
      //          << streamId << std::endl;
      waitingTaskHolder.doneWaiting(nullptr);
    } else {
      // wrap the exception in a try-catch block to let GDB "catch throw" break on it
      try {
        /*
        DPCT1009:69: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
        */
        auto error = "cudaGetErrorName not supported" /*cudaGetErrorName(status)*/;
        /*
        DPCT1009:70: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
        */
        auto message = "cudaGetErrorString not supported" /*cudaGetErrorString(status)*/;
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
      dpct::dev_mgr::instance().select_device(currentDevice_);
      stream_;
    }

    ScopedContextBase::ScopedContextBase(const ProductBase& data) : currentDevice_(data.device()) {
      dpct::dev_mgr::instance().select_device(currentDevice_);
      if (data.mayReuseStream()) {
        stream_ = data.streamPtr();
      } else {
        // FIXME obtain the queue from the device
        stream_;
      }
    }

    ScopedContextBase::ScopedContextBase(sycl::device device, sycl::queue stream)
        : currentDevice_(device), stream_(std::move(stream)) {
      dpct::dev_mgr::instance().select_device(currentDevice_);
    }

    ////////////////////

    void ScopedContextGetterBase::synchronizeStreams(int dataDevice,
                                                     sycl::queue* dataStream,
                                                     bool available,
                                                     sycl::event dataEvent) {
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
          dataEvent.wait();
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(sycl::device device, sycl::queue stream) {
      std::async([&]() {
                   stream->wait(); cudaScopedContextCallback(stream, 0, new CallbackData{waitingTaskHolder_, device});
                 })
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() {
    holderHelper_.enqueueCallback(device(), stream());
    if (contextState_) {
      contextState_->set(device(), std::move(streamPtr()));
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
    DPCT1012:76: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
    */
    cudaEventRecord(event_.get(), stream());
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(device(), stream()); }
}  // namespace cms::cuda
