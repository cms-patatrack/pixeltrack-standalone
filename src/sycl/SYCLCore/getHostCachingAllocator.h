#ifndef SYCLCore_getHostCachingAllocator_h
#define SYCLCore_getHostCachingAllocator_h

#include "SYCLCore/getPlatformIndex.h"
#include "SYCLCore/AllocatorConfig.h"
#include "SYCLCore/CachingAllocator.h"

namespace cms::sycltools {

  namespace detail {
    inline auto allocate_host_allocators() {
      using Allocator = cms::sycltools::CachingAllocator;
      auto const& platforms = enumeratePlatforms();
      auto const size = platforms.size();

      // allocate the storage for the objects
      auto ptr = std::allocator<Allocator>().allocate(size);

      // construct the objects in the storage
      for (size_t index = 0; index < size; ++index) {
        new (ptr + index) Allocator(platforms[index],
                                    true,  //isHost
                                    config::binGrowth,
                                    config::minBin,
                                    config::maxBin,
                                    config::maxCachedBytes,
                                    config::maxCachedFraction,
                                    config::allocator_policy,  // reuseSameQueueAllocations
                                    false);                    // debug
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

  inline CachingAllocator& getHostCachingAllocator(sycl::queue const& stream) {
    static auto allocators = detail::allocate_host_allocators();

    size_t const index = getPlatformIndex(stream.get_device().get_platform());
    assert(index < cms::sycltools::enumeratePlatforms().size());

    // the public interface is thread safe
    return allocators[index];
  }

}  // namespace cms::sycltools

#endif  // SYCLCore_getHostCachingAllocator_h