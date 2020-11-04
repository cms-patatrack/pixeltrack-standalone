#ifndef CUDACore_GenericCachingAllocator_h
#define CUDACore_GenericCachingAllocator_h

#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#include "deviceAllocatorStatus.h"

// Inspired by cub::CachingDeviceAllocator

/*
struct CUDAHostTraits {
  using DeviceType = int;
  using QueueType = cudaStream_t;
  using EventType = cudaEvent_t;
  struct Dummy {};

  constexpr DeviceType kInvalidDevice = -1;

  static DeviceType currentDevice() {
    return cms::cuda::currentDevice();
  }

  static Dummy setDevice(DeviceType device) {
    return {};
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

  static bool eventWorkHasCompleted(EventType e) {
    return cms::cuda::eventWorkHasCompleted(e);
  }

  static EventType createEvent() {
    EventType e;
    cudaCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return e;
  }

  static void destroyEvent(EventType e) {
    cudaCheck(cudaEventDestroy(e));
  }

  static EventType recreateEvent(EventType e, DeviceType prev, DeviceType next) {
    cudaCheck(cudaSetDevice(prev));
    destroyEvent(e);
    cudaCheck(cudaSetDevice(next));
    return createEvent();
  }

  static EventType recordEvent(EventType e, QueueType queue) {
    cudaCheck(cudaEventRecord(e, queue));
  }

  static std::ostream& printDevice(std::ostream& os, DeviceType dev) {
    os << "Host";
    return os;
  }

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

  static void free(void* ptr) {
    cudaCheck(cudaFreeHost(ptr));
  }
}
*/

namespace allocator {
  inline unsigned int intPow(unsigned int base, unsigned int exp) {
    unsigned int ret = 1;
    while (exp > 0) {
      if (exp & 1) {
        ret = ret * base;
      }
      base = base * base;
      exp = exp >> 1;
    }
    return ret;
  }

  // return (power, roundedBytes)
  inline std::tuple<unsigned int, size_t> nearestPowerOf(unsigned int base, size_t value) {
    unsigned int power = 0;
    size_t roundedBytes = 1;
    if (value * base < value) {
      // Overflow
      power = sizeof(size_t) * 8;
      roundedBytes = size_t(0) - 1;
    } else {
      while (roundedBytes < value) {
        roundedBytes *= base;
        ++power;
      }
    }

    return std::tuple(power, roundedBytes);
  }
}  // namespace allocator

template <typename Traits>
class GenericCachingAllocator {
public:
  using DeviceType = typename Traits::DeviceType;
  using QueueType = typename Traits::QueueType;
  using EventType = typename Traits::EventType;

  using TotalBytes = cms::cuda::allocator::TotalBytes;
  using DeviceCachedBytes = std::map<DeviceType, TotalBytes>;

  explicit GenericCachingAllocator(
      unsigned int binGrowth, unsigned int minBin, unsigned int maxBin, size_t maxCachedBytes, bool debug)
      : cachedBlocks_(&BlockDescriptor::SizeCompare),
        liveBlocks_(&BlockDescriptor::PtrCompare),
        minBinBytes_(allocator::intPow(binGrowth, minBin)),
        maxBinBytes_(allocator::intPow(binGrowth, maxBin)),
        maxCachedBytes_(maxCachedBytes),
        binGrowth_(binGrowth),
        minBin_(minBin),
        maxBin_(maxBin),
        debug_(debug) {}
  ~GenericCachingAllocator() { freeAllCached(); }

  // Cache allocation status (for monitoring purposes)
  DeviceCachedBytes cacheStatus() const {
    std::scoped_lock lock(mutex_);
    return cachedBytes_;
  }

  // Allocate given number of bytes on the current device associated to queue
  void* allocate(DeviceType device, size_t bytes, QueueType queue) {
    if (bytes > maxBinBytes_) {
      throw std::runtime_error("Requested allocation size " + std::to_string(bytes) +
                               " bytes is too large for the caching allocator with maximum bin " +
                               std::to_string(maxBinBytes_) +
                               " bytes. You might want to increase the maximum bin size");
    }

    // Create a block descriptor for the requested allocation
    BlockDescriptor searchKey;
    searchKey.bytesRequested = bytes;
    searchKey.device = device;
    searchKey.associatedQueue = queue;
    if (bytes < minBinBytes_) {
      searchKey.bin = minBin_;
      searchKey.bytes = minBinBytes_;
    } else {
      std::tie(searchKey.bin, searchKey.bytes) = allocator::nearestPowerOf(binGrowth_, bytes);
    }

    // Try to re-use cached block
    searchKey.ptr = tryReuseCachedBlock(searchKey);

    // allocate if necessary
    if (searchKey.ptr == nullptr) {
      auto scopedSetDevice = Traits::setDevice(device);

      searchKey.ptr = Traits::tryAllocate(searchKey.bytes);
      if (searchKey.ptr == nullptr) {
        // The allocation attempt failed: free all cached blocks on device and retry
        if (debug_) {
          std::cout << "\t";
          Traits::printDevice(std::cout, device)
              << " failed to allocate " << searchKey.bytes << " bytes for queue " << searchKey.associatedQueue
              << ", retrying after freeing cached allocations" << std::endl;
        }

        freeCachedBlocksOnDevice(device);

        searchKey.ptr = Traits::allocate(searchKey.bytes);
      }

      searchKey.readyEvent = Traits::createEvent();

      {
        std::scoped_lock lock(mutex_);
        liveBlocks_.insert(searchKey);
        cachedBytes_[device].live += searchKey.bytes;
        cachedBytes_[device].liveRequested += searchKey.bytesRequested;
      }

      if (debug_) {
        std::cout << "\t";
        Traits::printDevice(std::cout, device)
            << " allocated new block at " << searchKey.ptr << " (" << searchKey.bytes << " bytes associated with queue "
            << searchKey.associatedQueue << ", event " << searchKey.readyEvent << "." << std::endl;
      }
    }

    if (debug_) {
      std::cout << "\t\t" << cachedBlocks_.size() << " available blocks cached (" << cachedBytes_[device].free
                << " bytes), " << liveBlocks_.size() << " live blocks outstanding (" << cachedBytes_[device].live
                << " bytes)." << std::endl;
    }

    return searchKey.ptr;
  }

  // Frees an allocation on a given device
  void free(DeviceType device, void* ptr) {
    bool recache = false;
    BlockDescriptor searchKey;
    searchKey.device = device;
    searchKey.ptr = ptr;

    auto scopedSetDevice = Traits::setDevice(device);

    {
      std::scoped_lock lock(mutex_);

      auto iBlock = liveBlocks_.find(searchKey);
      if (iBlock == liveBlocks_.end()) {
        std::stringstream ss;
        ss << "Trying to free a non-live block at " << ptr;
        throw std::runtime_error(ss.str());
      }
      searchKey = *iBlock;
      liveBlocks_.erase(iBlock);
      cachedBytes_[device].live -= searchKey.bytes;
      cachedBytes_[device].liveRequested -= searchKey.bytesRequested;

      recache = (cachedBytes_[device].free + searchKey.bytes <= maxCachedBytes_);
      if (recache) {
        cachedBlocks_.insert(searchKey);
        cachedBytes_[device].free += searchKey.bytes;

        if (debug_) {
          std::cout << "\t";
          Traits::printDevice(std::cout, device)
              << " returned " << searchKey.bytes << " bytes at " << ptr << " from associated queue "
              << searchKey.associatedQueue << " , event " << searchKey.readyEvent << " .\n\t\t " << cachedBlocks_.size()
              << " available "
                 "blocks cached ("
              << cachedBytes_[device].free << " bytes), " << liveBlocks_.size() << " live blocks outstanding. ("
              << cachedBytes_[device].live << " bytes)" << std::endl;
        }
      }

      if (recache) {
        Traits::recordEvent(searchKey.readyEvent, searchKey.associatedQueue);
      }
    }

    if (not recache) {
      Traits::free(ptr);
      Traits::destroyEvent(searchKey.readyEvent);
      if (debug_) {
        std::cout << "\t";
        Traits::printDevice(std::cout, device)
            << " freed " << searchKey.bytes << " bytes at " << ptr << " from associated queue "
            << searchKey.associatedQueue << ", event " << searchKey.readyEvent << ".\n\t\t  " << cachedBlocks_.size()
            << " available "
               "blocks cached ("
            << cachedBytes_[device].free << " bytes), " << liveBlocks_.size() << " live blocks ("
            << cachedBytes_[device].live << " bytes) outstanding." << std::endl;
      }
    }
  }

private:
  struct BlockDescriptor {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t bytesRequested = 0;  // for monitoring only
    unsigned int bin = 0;
    DeviceType device = Traits::kInvalidDevice;
    QueueType associatedQueue;
    EventType readyEvent;

    static bool PtrCompare(BlockDescriptor const& a, BlockDescriptor const& b) {
      if (a.device == b.device)
        return a.ptr < b.ptr;
      return a.device < b.device;
    }

    static bool SizeCompare(BlockDescriptor const& a, BlockDescriptor const& b) {
      if (a.device == b.device)
        return a.bytes < b.bytes;
      return a.device < b.device;
    }
  };

  void* tryReuseCachedBlock(BlockDescriptor& searchKey) {
    std::scoped_lock lock(mutex_);

    // Iterate through the range of cached blocks on the same device in the same bin
    for (auto iBlock = cachedBlocks_.lower_bound(searchKey);
         iBlock != cachedBlocks_.end() and Traits::canReuseInDevice(searchKey.device, iBlock->device) and
         iBlock->bin == searchKey.bin;
         ++iBlock) {
      if (Traits::canReuseInQueue(searchKey.associatedQueue, iBlock->associatedQueue) or
          Traits::eventWorkHasCompleted(iBlock->readyEvent)) {
        // Reuse existing cache block. Insert into live blocks.
        auto device = searchKey.device;
        auto queue = searchKey.associatedQueue;
        searchKey = *iBlock;
        searchKey.associatedQueue = queue;

        if (searchKey.device != device) {
          searchKey.readyEvent = Traits::recreateEvent(searchKey.readyEvent, searchKey.device, device);
          searchKey.device = device;
        }

        liveBlocks_.insert(searchKey);

        cachedBytes_[device].free -= searchKey.bytes;
        cachedBytes_[device].live += searchKey.bytes;
        cachedBytes_[device].live += searchKey.bytesRequested;

        if (debug_) {
          std::cout << "\t";
          Traits::printDevice(std::cout, device)
              << " reused cached block at " << searchKey.ptr << " (" << searchKey.bytes << "bytes) for queue "
              << searchKey.associatedQueue << ", event " << searchKey.readyEvent
              << " (previously "
                 "associated with stream "
              << iBlock->associatedQueue << " , event " << iBlock->readyEvent << ")." << std::endl;
        }

        cachedBlocks_.erase(iBlock);
        return searchKey.ptr;
      }
    }

    return nullptr;
  }

  void freeCachedBlocksOnDevice(DeviceType device) {
    std::scoped_lock lock(mutex_);

    BlockDescriptor freeKey;
    freeKey.device = device;
    for (auto iBlock = cachedBlocks_.lower_bound(freeKey);
         iBlock != cachedBlocks_.end() and iBlock->device == device;) {
      Traits::free(iBlock->ptr);
      Traits::destroyEvent(iBlock->readyEvent);
      cachedBytes_[device].free -= iBlock->bytes;

      if (debug_) {
        std::cout << "\t";
        Traits::printDevice(std::cout, device)
            << " freed " << iBlock->bytes << " bytes.\n\t\t  " << cachedBlocks_.size() << " available blocks cached ("
            << cachedBytes_[device].free << " bytes), " << liveBlocks_.size()
            << " live blocks "
               "("
            << cachedBytes_[device].live << " bytes) outstanding." << std::endl;
      }

      iBlock = cachedBlocks_.erase(iBlock);
    }
  }

  void freeAllCached() {
    std::scoped_lock lock(mutex_);

    while (not cachedBlocks_.empty()) {
      auto iBlock = cachedBlocks_.begin();
      auto scopedSetDevice = Traits::setDevice(iBlock->device);
      Traits::free(iBlock->ptr);
      Traits::destroyEvent(iBlock->readyEvent);
      cachedBytes_[iBlock->device].free -= iBlock->bytes;

      if (debug_) {
        std::cout << "\t";
        Traits::printDevice(std::cout, iBlock->device)
            << " freed " << iBlock->bytes << " bytes.\n\t\t  " << (cachedBlocks_.size()-1) << " available blocks cached ("
            << cachedBytes_[iBlock->device].free << " bytes), " << liveBlocks_.size() << " live blocks ("
            << cachedBytes_[iBlock->device].live << " bytes) outstanding." << std::endl;
      }

      cachedBlocks_.erase(iBlock);
    }
  }

  using Compare = typename std::add_pointer<bool(BlockDescriptor const&, BlockDescriptor const&)>::type;
  using CachedBlocks = std::multiset<BlockDescriptor, Compare>;  // ordered by size
  using BusyBlocks = std::multiset<BlockDescriptor, Compare>;    // ordered by ptr

  mutable std::mutex mutex_;

  DeviceCachedBytes cachedBytes_;
  CachedBlocks cachedBlocks_;  // Set of cached device allocations available for reuse
  BusyBlocks liveBlocks_;      // Set of live device allocations currently in use

  size_t const minBinBytes_;
  size_t const maxBinBytes_;
  size_t const maxCachedBytes_;  // Maximum aggregate cached bytes per device

  unsigned int const binGrowth_;  // Geometric growth factor for bin-sizes
  unsigned int const minBin_;
  unsigned int const maxBin_;

  bool const debug_;
};

#endif
