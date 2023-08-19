#include <cassert>
#include <limits>

#include <hip/hip_runtime.h>

#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::hip::allocator::binGrowth, cms::hip::allocator::maxBin);
}

namespace cms::hip {
  void *allocate_device(int dev, size_t nbytes, hipStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
#if HIP_VERSION >= 50200000
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(hipMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(hipMalloc(&ptr, nbytes));
    }
    return ptr;
  }

  void free_device(int device, void *ptr, hipStream_t stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
#if HIP_VERSION >= 50200000
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(hipFreeAsync(ptr, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(hipFree(ptr));
    }
  }

}  // namespace cms::hip
