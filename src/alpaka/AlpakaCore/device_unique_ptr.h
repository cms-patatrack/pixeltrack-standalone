#ifndef HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h

#include <memory>
#include <type_traits>

#include "AlpakaCore/host_unique_ptr.h"

namespace cms {
  namespace alpakatools {
    namespace device {
      namespace impl {
        template <typename TData>
        class DeviceDeleter {
        public:
          DeviceDeleter(ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> buffer) : buf{std::move(buffer)} {}

          void operator()(void* d_ptr) {
            if constexpr (allocator::policy == allocator::Policy::Caching) {
              if (d_ptr) {
                allocator::getCachingDeviceAllocator().DeviceFree(buf);
              }
            }
          }

        private:
          ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> buf;
        };
      }  // namespace impl
      template <typename TData>
      using unique_ptr =
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
          std::unique_ptr<TData,
                          impl::DeviceDeleter<
                              std::conditional_t<allocator::policy == allocator::Policy::Caching, std::byte, TData>>>;
#else
          host::unique_ptr<TData>;
#endif
    }  // namespace device

    template <typename TData>
    auto make_device_unique(const alpaka_common::Extent& extent) {
      const auto& device = ALPAKA_ACCELERATOR_NAMESPACE::device;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      if constexpr (allocator::policy == allocator::Policy::Caching) {
        const alpaka_common::Extent nbytes = alpakatools::nbytesFromExtent<TData>(extent);
        if (nbytes > maxAllocationSize) {
          throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                   " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
        }
        auto buf = allocator::getCachingDeviceAllocator().DeviceAllocate(nbytes, device);
        void* d_ptr = alpaka::getPtrNative(buf);
        return typename device::unique_ptr<TData>{reinterpret_cast<TData*>(d_ptr),
                                                  device::impl::DeviceDeleter<std::byte>{buf}};
      } else {
        auto buf = alpaka::allocBuf<TData, alpaka_common::Idx>(device, extent);
#if CUDA_VERSION >= 11020
        if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
          alpaka::prepareForAsyncCopy(buf);
        }
#endif
        TData* d_ptr = alpaka::getPtrNative(buf);
        return typename device::unique_ptr<TData>{d_ptr, device::impl::DeviceDeleter<TData>{buf}};
      }
#else
      return make_host_unique<TData>(extent);
#endif
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
