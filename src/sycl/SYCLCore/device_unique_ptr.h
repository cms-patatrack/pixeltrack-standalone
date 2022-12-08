#ifndef HeterogeneousCore_SYCLUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_SYCLUtilities_interface_device_unique_ptr_h

#include <functional>
#include <memory>
#include <optional>

#include <CL/sycl.hpp>
#include "SYCLCore/getDeviceCachingAllocator.h"

namespace cms {
  namespace sycltools {
    namespace device {
      namespace impl {
        // Additional layer of types to distinguish from host::unique_ptr
        class DeviceDeleter {
        public:
          DeviceDeleter() = default;  // for edm::Wrapper
          DeviceDeleter(sycl::queue stream) : stream_{stream} {}

          void operator()(void* ptr) {
            if (stream_) {
              CachingAllocator& allocator = getDeviceCachingAllocator(*stream_);
              allocator.free(ptr);
            }
          }

        private:
          std::optional<sycl::queue> stream_;
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;

      namespace impl {
        template <typename T>
        struct make_device_unique_selector {
          using non_array = cms::sycltools::device::unique_ptr<T>;
        };
        template <typename T>
        struct make_device_unique_selector<T[]> {
          using unbounded_array = cms::sycltools::device::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_device_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace device

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique(sycl::queue const& stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      CachingAllocator& allocator = getDeviceCachingAllocator(stream);
      void* mem = allocator.allocate(sizeof(T), stream);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T*>(mem),
                                                                              device::impl::DeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(
        size_t n, sycl::queue const& stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      CachingAllocator& allocator = getDeviceCachingAllocator(stream);
      void* mem = allocator.allocate(n * sizeof(element_type), stream);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type*>(mem), device::impl::DeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique(Args&&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique_uninitialized(
        sycl::queue const& stream) {
      CachingAllocator& allocator = getDeviceCachingAllocator(stream);
      void* mem = allocator.allocate(sizeof(T), stream);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T*>(mem),
                                                                              device::impl::DeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique_uninitialized(
        size_t n, sycl::queue const& stream) {
      using element_type = typename std::remove_extent<T>::type;
      CachingAllocator& allocator = getDeviceCachingAllocator(stream);
      void* mem = allocator.allocate(n * sizeof(element_type), stream);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type*>(mem), device::impl::DeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique_uninitialized(Args&&...) =
        delete;
  }  // namespace sycltools
}  // namespace cms

#endif  //HeterogeneousCore_SYCLUtilities_interface_device_unique_ptr_h