#ifndef HeterogeneousCore_CUDACore_src_getCachingManagedAllocator
#define HeterogeneousCore_CUDACore_src_getCachingManagedAllocator

#include "CUDACore/cudaCheck.h"
#include "CachingManagedAllocator.h"

#include "getCachingDeviceAllocator.h"

namespace cms::cuda::allocator {
  inline notcub::CachingManagedAllocator& getCachingManagedAllocator() {
    // the public interface is thread safe
    static notcub::CachingManagedAllocator allocator{binGrowth,
                                                     minBin,
                                                     maxBin,
                                                     minCachedBytes(),
                                                     false,  // do not skip cleanup
                                                     debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
