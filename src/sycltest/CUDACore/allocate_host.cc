#include <limits>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "CUDACore/allocate_host.h"
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
      allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream);
    } else {
      ptr = (void *)sycl::malloc_host(nbytes, dpct::get_default_queue());
    }
    return ptr;
  }

  void free_host(void *ptr) {
    if constexpr (allocator::useCaching) {
      allocator::getCachingHostAllocator().HostFree(ptr);
    } else {
      sycl::free(ptr, dpct::get_default_queue());
    }
  }

}  // namespace cms::cuda
