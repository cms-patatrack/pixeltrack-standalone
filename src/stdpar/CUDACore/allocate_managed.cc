#include <limits>

#include "CUDACore/allocate_managed.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingManagedAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_managed(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::useCaching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingManagedAllocator().ManagedAllocate(&ptr, nbytes, stream));
    } else {
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
    }
    return ptr;
  }

  void free_managed(void *ptr) {
    if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingManagedAllocator().ManagedFree(ptr));
    } else {
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
