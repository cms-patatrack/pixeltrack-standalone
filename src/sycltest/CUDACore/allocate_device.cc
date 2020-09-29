#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <limits>

#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_device(int dev, size_t nbytes, sycl::queue *stream) {
    void *ptr = nullptr;
    if constexpr (allocator::useCaching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      /*
      DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((ptr = (void *)sycl::malloc_device(nbytes, dpct::get_default_queue()), 0));
    }
    return ptr;
  }

  void free_device(int device, void *ptr) {
    if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      /*
      DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((sycl::free(ptr, dpct::get_default_queue()), 0));
    }
  }

}  // namespace cms::cuda
