#ifndef HeterogeneousCore_AlpakaCore_src_getCachingDeviceAllocator
#define HeterogeneousCore_AlpakaCore_src_getCachingDeviceAllocator

#include <iomanip>
#include <iostream>

#include "AlpakaCore/CachingDeviceAllocator.h"

namespace cms::alpakatools::allocator {
  // Use caching or not
  enum class Policy { Synchronous = 0, Asynchronous = 1, Caching = 2 };
#ifndef ALPAKA_DISABLE_CACHING_ALLOCATOR
  constexpr Policy policy = Policy::Caching;
#elif CUDA_VERSION >= 11020 && !defined ALPAKA_DISABLE_ASYNC_ALLOCATOR
  constexpr Policy policy = Policy::Asynchronous;
#else
  constexpr Policy policy = Policy::Synchronous;
#endif
  // Growth factor (bin_growth in CachingDeviceAllocator
  constexpr unsigned int binGrowth = 2;
  // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in CacingDeviceAllocator
  constexpr unsigned int minBin = 8;
  // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in CachingDeviceAllocator). Note that unlike in allocator, allocations larger than binGrowth^maxBin are set to fail.
  constexpr unsigned int maxBin = 30;
  // Total storage for the allocator. 0 means no limit.
  constexpr size_t maxCachedBytes = 0;
  // Fraction of total device memory taken for the allocator. In case there are multiple devices with different amounts of memory, the smallest of them is taken. If maxCachedBytes is non-zero, the smallest of them is taken.
  constexpr double maxCachedFraction = 0.8;
  constexpr bool debug = false;

  inline size_t minCachedBytes() {
    size_t ret = std::numeric_limits<size_t>::max();
    const auto devices{alpaka::getDevs<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>()};
    for (const auto& device : devices) {
      const size_t freeMemory{alpaka::getFreeMemBytes(device)};
      ret = std::min(ret, static_cast<size_t>(maxCachedFraction * freeMemory));
    }
    if (maxCachedBytes > 0) {
      ret = std::min(ret, maxCachedBytes);
    }
    return ret;
  }

  inline CachingDeviceAllocator& getCachingDeviceAllocator() {
    if (debug) {
      std::cout << "CachingDeviceAllocator settings\n"
                << "  bin growth " << binGrowth << "\n"
                << "  min bin    " << minBin << "\n"
                << "  max bin    " << maxBin << "\n"
                << "  resulting bins:\n";
      for (auto bin = minBin; bin <= maxBin; ++bin) {
        auto binSize = CachingDeviceAllocator::IntPow(binGrowth, bin);
        if (binSize >= (1 << 30) and binSize % (1 << 30) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 30) << " GB\n";
        } else if (binSize >= (1 << 20) and binSize % (1 << 20) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 20) << " MB\n";
        } else if (binSize >= (1 << 10) and binSize % (1 << 10) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 10) << " kB\n";
        } else {
          std::cout << "    " << std::setw(9) << binSize << " B\n";
        }
      }
      std::cout << "  maximum amount of cached memory: " << (minCachedBytes() >> 20) << " MB\n";
    }

    // the public interface is thread safe
    static CachingDeviceAllocator allocator{binGrowth, minBin, maxBin, minCachedBytes(), debug};
    return allocator;
  }
}  // namespace cms::alpakatools::allocator

#endif
