#include "CUDACore/EDGetterContextBase.h"
#include "CUDACore/cudaCheck.h"

namespace cms::cuda::impl {
  void EDGetterContextBase::synchronizeStreams(int dataDevice,
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
}
