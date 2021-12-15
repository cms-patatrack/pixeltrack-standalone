#ifndef HeterogeneousCore_CUDACore_src_getCachingHostAllocator
#define HeterogeneousCore_CUDACore_src_getCachingHostAllocator

#include <iomanip>
#include <iostream>

#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace cms::cuda::allocator {
  struct HostTraits {
    using DeviceType = int;
    using QueueType = cudaStream_t;
    using EventType = cudaEvent_t;

    static constexpr DeviceType kInvalidDevice = -1;
    static constexpr DeviceType kHostDevice = 0;

    static DeviceType currentDevice() { return cms::cuda::currentDevice(); }

    static DeviceType memoryDevice(DeviceType deviceEvent) {
      // For host allocator the "current device" refers to the GPU for
      // which the event has been recorded. The "device" where the
      // memory is allocated on is always 0.
      return kHostDevice;
    }

    static bool canReuseInDevice(DeviceType a, DeviceType b) {
      // Pinned host memory can be reused in any device, but in case of
      // changes the event must be re-created
      return true;
    }

    static bool canReuseInQueue(QueueType a, QueueType b) {
      // For pinned host memory a freed block without completed event
      // can not be re-used even for operations in the same queue
      return false;
    }

    template <typename C>
    static bool deviceCompare(DeviceType a_dev, DeviceType b_dev, C&& compare) {
      return compare();
    }

    static bool eventWorkHasCompleted(EventType e) { return cms::cuda::eventWorkHasCompleted(e); }

    static EventType createEvent() {
      EventType e;
      cudaCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      return e;
    }

    static void destroyEvent(EventType e) { cudaCheck(cudaEventDestroy(e)); }

    static EventType recreateEvent(EventType e, DeviceType prev, DeviceType next) {
      cudaCheck(cudaSetDevice(prev));
      destroyEvent(e);
      cudaCheck(cudaSetDevice(next));
      return createEvent();
    }

    static EventType recordEvent(EventType e, QueueType queue) {
      cudaCheck(cudaEventRecord(e, queue));
      return e;
    }

    struct DevicePrinter {
      static void write(std::ostream& os) { os << "Host"; }
    };

    static DevicePrinter printDevice(DeviceType dev) { return DevicePrinter(); }

    static void* allocate(size_t bytes) {
      void* ptr;
      cudaCheck(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
      return ptr;
    }

    static void* tryAllocate(size_t bytes) {
      void* ptr;
      auto error = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
      if (error == cudaErrorMemoryAllocation) {
        return nullptr;
      }
      cudaCheck(error);
      return ptr;
    }

    static void free(void* ptr) { cudaCheck(cudaFreeHost(ptr)); }
  };

  inline std::ostream& operator<<(std::ostream& os, HostTraits::DevicePrinter const& pr) {
    pr.write(os);
    return os;
  }

  using CachingHostAllocator = GenericCachingAllocator<HostTraits>;

  inline CachingHostAllocator& getCachingHostAllocator() {
    if (debug) {
      std::cout << "cub::CachingHostAllocator settings\n"
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
    static CachingHostAllocator allocator{binGrowth, minBin, maxBin, minCachedBytes(), debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
