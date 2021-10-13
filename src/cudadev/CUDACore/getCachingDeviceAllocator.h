#ifndef HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator
#define HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator

#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include "CUDACore/cudaCheck.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/eventWorkHasCompleted.h"
#include "CUDACore/GenericCachingAllocator.h"

namespace cms::cuda::allocator {
  // Use caching or not
  enum class Policy { Synchronous = 0, Asynchronous = 1, Caching = 2 };
#ifndef CUDADEV_DISABLE_CACHING_ALLOCATOR
  constexpr Policy policy = Policy::Caching;
#elif CUDA_VERSION >= 11020 && !defined CUDADEV_DISABLE_ASYNC_ALLOCATOR
  constexpr Policy policy = Policy::Asynchronous;
#else
  constexpr Policy policy = Policy::Synchronous;
#endif
  // Growth factor (bin_growth in cub::CachingDeviceAllocator
  constexpr unsigned int binGrowth = 2;
  // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CacingDeviceAllocator
  constexpr unsigned int minBin = 8;
  // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
  constexpr unsigned int maxBin = 30;
  // Total storage for the allocator. 0 means no limit.
  constexpr size_t maxCachedBytes = 0;
  // Fraction of total device memory taken for the allocator. In case there are multiple devices with different amounts of memory, the smallest of them is taken. If maxCachedBytes is non-zero, the smallest of them is taken.
  constexpr double maxCachedFraction = 0.8;
  constexpr bool debug = false;

  inline size_t minCachedBytes() {
    size_t ret = std::numeric_limits<size_t>::max();
    int currentDevice;
    cudaCheck(cudaGetDevice(&currentDevice));
    const int numberOfDevices = deviceCount();
    for (int i = 0; i < numberOfDevices; ++i) {
      size_t freeMemory, totalMemory;
      cudaCheck(cudaSetDevice(i));
      cudaCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
      ret = std::min(ret, static_cast<size_t>(maxCachedFraction * freeMemory));
    }
    cudaCheck(cudaSetDevice(currentDevice));
    if (maxCachedBytes > 0) {
      ret = std::min(ret, maxCachedBytes);
    }
    return ret;
  }

  struct DeviceTraits {
    using DeviceType = int;
    using QueueType = cudaStream_t;
    using EventType = cudaEvent_t;

    static constexpr DeviceType kInvalidDevice = -1;

    static DeviceType currentDevice() { return cms::cuda::currentDevice(); }

    static DeviceType memoryDevice(DeviceType deviceEvent) {
      // For device allocator the device where the memory is allocated
      // on is the same as the device where the event is recorded on.
      return deviceEvent;
    }

    static bool canReuseInDevice(DeviceType a, DeviceType b) { return a == b; }

    static bool canReuseInQueue(QueueType a, QueueType b) { return a == b; }

    template <typename C>
    static bool deviceCompare(DeviceType a_dev, DeviceType b_dev, C&& compare) {
      if (a_dev == b_dev) {
        return compare();
      }
      return a_dev < b_dev;
    }

    static bool eventWorkHasCompleted(EventType e) { return cms::cuda::eventWorkHasCompleted(e); }

    static EventType createEvent() {
      EventType e;
      cudaCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      return e;
    }

    static void destroyEvent(EventType e) { cudaCheck(cudaEventDestroy(e)); }

    static EventType recreateEvent(EventType e, DeviceType prev, DeviceType next) {
      throw std::runtime_error("CUDADeviceTraits::recreateEvent() should never be called");
    }

    static void recordEvent(EventType e, QueueType queue) { cudaCheck(cudaEventRecord(e, queue)); }

    struct DevicePrinter {
      DevicePrinter(DeviceType device) : device_(device) {}
      void write(std::ostream& os) const { os << "Device " << device_; }
      DeviceType device_;
    };

    static DevicePrinter printDevice(DeviceType device) { return DevicePrinter(device); }

    static void* allocate(size_t bytes) {
      void* ptr;
      cudaCheck(cudaMalloc(&ptr, bytes));
      return ptr;
    }

    static void* tryAllocate(size_t bytes) {
      void* ptr;
      auto error = cudaMalloc(&ptr, bytes);
      if (error == cudaErrorMemoryAllocation) {
        return nullptr;
      }
      cudaCheck(error);
      return ptr;
    }

    static void free(void* ptr) { cudaCheck(cudaFree(ptr)); }
  };

  inline std::ostream& operator<<(std::ostream& os, DeviceTraits::DevicePrinter const& pr) {
    pr.write(os);
    return os;
  }

  using CachingDeviceAllocator = GenericCachingAllocator<DeviceTraits>;

  inline CachingDeviceAllocator& getCachingDeviceAllocator() {
    if (debug) {
      std::cout << "cub::CachingDeviceAllocator settings\n"
                << "  bin growth " << binGrowth << "\n"
                << "  min bin    " << minBin << "\n"
                << "  max bin    " << maxBin << "\n"
                << "  resulting bins:\n";
      for (auto bin = minBin; bin <= maxBin; ++bin) {
        auto binSize = ::allocator::intPow(binGrowth, bin);
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
}  // namespace cms::cuda::allocator

#endif
