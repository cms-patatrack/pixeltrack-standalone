#include <cassert>
#include <limits>

#include <cuda_runtime.h>

#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize = allocator::intPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_device(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      ptr = allocator::getCachingDeviceAllocator().allocate(nbytes, stream);
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      cudaCheck(cudaMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      cudaCheck(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

  void free_device(void *ptr, cudaStream_t stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingDeviceAllocator().free(ptr);
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      cudaCheck(cudaFreeAsync(ptr, stream));
#endif
    } else {
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
