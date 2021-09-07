#ifndef HeterogeneousCore_CUDAUtilities_deviceAllocatorStatus_h
#define HeterogeneousCore_CUDAUtilities_deviceAllocatorStatus_h

#include <map>

namespace cms {
  namespace cuda {
    namespace allocator {
      struct TotalBytes {
        size_t free = 0;
        size_t live = 0;
        size_t liveRequested = 0;
      };
      /// Map type of device ordinals to the number of cached bytes cached by each device
      using GpuCachedBytes = std::map<int, TotalBytes>;
    }  // namespace allocator

    allocator::GpuCachedBytes deviceAllocatorStatus();
  }  // namespace cuda
}  // namespace cms

#endif
