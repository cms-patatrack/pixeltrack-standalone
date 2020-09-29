#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <limits>

#include "CUDACore/allocate_host.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingHostAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_host(size_t nbytes, sycl::queue *stream) {
    void *ptr = nullptr;
    if constexpr (allocator::useCaching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream));
    } else {
      cudaCheck((ptr = (void *)sycl::malloc_host(nbytes, dpct::get_default_queue()), 0));
    }
    return ptr;
  }

  void free_host(void *ptr) {
    if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingHostAllocator().HostFree(ptr));
    } else {
      /*
      DPCT1003:68: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((sycl::free(ptr, dpct::get_default_queue()), 0));
    }
  }

}  // namespace cms::cuda
