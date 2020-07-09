#ifndef HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator
#define HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CachingDeviceAllocator.h"

namespace cms::cuda::allocator {
  // Use caching or not
  constexpr bool useCaching = true;
  // Growth factor (bin_growth in cub::CachingDeviceAllocator
  constexpr unsigned int binGrowth = 8;
  // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CacingDeviceAllocator
  constexpr unsigned int minBin = 1;
  // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
  constexpr unsigned int maxBin = 10;
  // Total storage for the allocator. 0 means no limit.
  constexpr size_t maxCachedBytes = 0;
  // Fraction of total device memory taken for the allocator. In case there are multiple devices with different amounts of memory, the smallest of them is taken. If maxCachedBytes is non-zero, the smallest of them is taken.
  constexpr double maxCachedFraction = 0.8;
  constexpr bool debug = false;

  inline size_t minCachedBytes() {
    size_t ret = std::numeric_limits<size_t>::max();
    int currentDevice;
    cudaCheck(currentDevice = dpct::dev_mgr::instance().current_device_id());
    const int numberOfDevices = deviceCount();
    for (int i = 0; i < numberOfDevices; ++i) {
      size_t freeMemory, totalMemory;
      /*
      DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((dpct::dev_mgr::instance().select_device(i), 0));
      cudaCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
      ret = std::min(ret, static_cast<size_t>(maxCachedFraction * freeMemory));
    }
    /*
    DPCT1003:40: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    cudaCheck((dpct::dev_mgr::instance().select_device(currentDevice), 0));
    if (maxCachedBytes > 0) {
      ret = std::min(ret, maxCachedBytes);
    }
    return ret;
  }

  inline notcub::CachingDeviceAllocator& getCachingDeviceAllocator() {
    // the public interface is thread safe
    static notcub::CachingDeviceAllocator allocator{binGrowth,
                                                    minBin,
                                                    maxBin,
                                                    minCachedBytes(),
                                                    false,  // do not skip cleanup
                                                    debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
