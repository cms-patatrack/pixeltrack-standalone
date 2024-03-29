#ifndef AlpakaCore_getDeviceCachingAllocator_h
#define AlpakaCore_getDeviceCachingAllocator_h

#include <optional>
#include <mutex>
#include <vector>

#include "AlpakaCore/AllocatorConfig.h"
#include "AlpakaCore/CachingAllocator.h"
#include "AlpakaCore/alpaka/devices.h"
#include "AlpakaCore/getDeviceIndex.h"

namespace cms::alpakatools {

  namespace detail {

    template <typename TDevice, typename TQueue>
    auto allocate_device_allocators() {
      using Allocator = CachingAllocator<TDevice, TQueue>;
      using Platform = alpaka::Platform<TDevice>;
      auto const& devices = cms::alpakatools::devices<Platform>();
      auto const size = devices.size();

      // allocate the storage for the objects
      auto ptr = std::allocator<Allocator>().allocate(size);

      // construct the objects in the storage
      for (size_t index = 0; index < size; ++index) {
        new (ptr + index) Allocator(devices[index],
                                    config::binGrowth,
                                    config::minBin,
                                    config::maxBin,
                                    config::maxCachedBytes,
                                    config::maxCachedFraction,
                                    true,    // reuseSameQueueAllocations
                                    false);  // debug
      }

      // use a custom deleter to destroy all objects and deallocate the memory
      auto deleter = [size](Allocator* ptr) {
        for (size_t i = size; i > 0; --i) {
          (ptr + i - 1)->~Allocator();
        }
        std::allocator<Allocator>().deallocate(ptr, size);
      };

      return std::unique_ptr<Allocator[], decltype(deleter)>(ptr, deleter);
    }

  }  // namespace detail

  template <typename TDevice, typename TQueue>
  inline CachingAllocator<TDevice, TQueue>& getDeviceCachingAllocator(TDevice const& device) {
    using Platform = alpaka::Platform<TDevice>;

    // initialise all allocators, one per device
    static auto allocators = detail::allocate_device_allocators<TDevice, TQueue>();

    size_t const index = getDeviceIndex(device);
    assert(index < cms::alpakatools::devices<Platform>().size());

    // the public interface is thread safe
    return allocators[index];
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_getDeviceCachingAllocator_h
