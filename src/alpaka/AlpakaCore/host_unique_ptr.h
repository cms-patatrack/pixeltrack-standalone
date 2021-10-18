#ifndef HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h

#include <memory>
#include <type_traits>

#include "AlpakaCore/getCachingHostAllocator.h"

namespace cms {
  namespace alpakatools {
    namespace host {
      namespace impl {
        template <typename TData>
        class HostDeleter {
        public:
          HostDeleter(alpaka_common::AlpakaHostBuf<TData> buffer) : buf{std::move(buffer)} {}

          void operator()(void* d_ptr) {
            if constexpr (allocator::policy == allocator::Policy::Caching) {
              if (d_ptr) {
                allocator::getCachingHostAllocator().HostFree(buf);
              }
            }
          }

        private:
          alpaka_common::AlpakaHostBuf<TData> buf;
        };
      }  // namespace impl

      template <typename TData>
      using unique_ptr = std::unique_ptr<
          TData,
          impl::HostDeleter<std::conditional_t<allocator::policy == allocator::Policy::Caching, std::byte, TData>>>;
    }  // namespace host

    inline constexpr size_t maxAllocationSize =
        allocator::CachingDeviceAllocator::IntPow(allocator::binGrowth, allocator::maxBin);

    // Allocate pinned host memory
    template <typename TData>
    typename host::unique_ptr<TData> make_host_unique(const alpaka_common::Extent& extent) {
      if constexpr (allocator::policy == allocator::Policy::Caching) {
        const alpaka_common::Extent nbytes = alpakatools::nbytesFromExtent<TData>(extent);
        if (nbytes > maxAllocationSize) {
          throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                   " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
        }
        auto buf = allocator::getCachingHostAllocator().HostAllocate(nbytes);
        void* d_ptr = alpaka::getPtrNative(buf);
        return
            typename host::unique_ptr<TData>{reinterpret_cast<TData*>(d_ptr), host::impl::HostDeleter<std::byte>{buf}};
      } else {
        auto buf = allocHostBuf<TData>(extent);
#if CUDA_VERSION >= 11020
        if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
          alpaka::prepareForAsyncCopy(buf);
        }
#endif
        TData* d_ptr = alpaka::getPtrNative(buf);
        return typename host::unique_ptr<TData>{d_ptr, host::impl::HostDeleter<TData>{buf}};
      }
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
