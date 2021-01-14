#include "KokkosCore/ProductBase.h"

#ifdef KOKKOS_ENABLE_CUDA
#include "CUDACore/cudaCheck.h"
#include "CUDACore/eventWorkHasCompleted.h"

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
#endif

namespace cms {
  namespace kokkos {
    namespace impl {
      ExecSpaceSpecificBase::~ExecSpaceSpecificBase() = default;

#ifdef KOKKOS_ENABLE_CUDA
      void ExecSpaceSpecific<Kokkos::Cuda>::enqueueCallback(edm::WaitingTaskWithArenaHolder holder) {
        cudaCheck(cudaStreamAddCallback(
            stream_.get(), cudaScopedContextCallback, new CallbackData{std::move(holder), device()}, 0));
      }

      void ExecSpaceSpecific<Kokkos::Cuda>::synchronizeWith(ExecSpaceSpecific const& other) {
        if (device() != other.device()) {
          // Eventually replace with prefetch to current device (assuming unified memory works)
          // If we won't go to unified memory, need to figure out something else...
          throw std::runtime_error("Handling data from multiple devices is not yet supported");
        }

        if (other.stream_.get() != stream_.get()) {
          // Different streams, need to synchronize
          if (not other.isAvailable()) {
            // Event not yet occurred, so need to add synchronization
            // here. Sychronization is done by making the CUDA stream to
            // wait for an event, so all subsequent work in the stream
            // will run only after the event has "occurred" (i.e. data
            // product became available).
            cudaCheck(cudaStreamWaitEvent(stream_.get(), other.event_.get(), 0),
                      "Failed to make a stream to wait for an event");
          }
        }
      }

      bool ExecSpaceSpecific<Kokkos::Cuda>::isAvailable() const {
        return cms::cuda::eventWorkHasCompleted(event_.get());
      }
#endif
    }  // namespace impl
  }    // namespace kokkos
}  // namespace cms
