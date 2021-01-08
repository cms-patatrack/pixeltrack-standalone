#include "CUDACore/deviceAllocatorStatus.h"

#include "getCachingDeviceAllocator.h"

namespace cms::hip {
  allocator::GpuCachedBytes deviceAllocatorStatus() { return allocator::getCachingDeviceAllocator().CacheStatus(); }
}  // namespace cms::hip
