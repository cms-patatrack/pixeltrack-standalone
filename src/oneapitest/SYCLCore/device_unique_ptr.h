#ifndef HeterogeneousCore_SYCLUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_SYCLUtilities_interface_device_unique_ptr_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <memory>
#include <functional>

#include "SYCLCore/allocate_device.h"

namespace cms {
  namespace sycl {
    namespace device {
      namespace impl {
        // Additional layer of types to distinguish from host::unique_ptr
        class DeviceDeleter {
        public:
          DeviceDeleter() = default;  // for edm::Wrapper
          DeviceDeleter(::sycl::queue stream) : stream_{stream} {}

          void operator()(void *ptr) {
            if (stream_) {
              free_device(*stream_, ptr);
            }
          }

        private:
          std::optional<::sycl::queue> stream_;
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;

      namespace impl {
        template <typename T>
        struct make_device_unique_selector {
          using non_array = cms::sycl::device::unique_ptr<T>;
        };
        template <typename T>
        struct make_device_unique_selector<T[]> {
          using unbounded_array = cms::sycl::device::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_device_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace device

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique(::sycl::queue stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      void *mem = allocate_device(stream, sizeof(T));
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(size_t n,
                                                                                              ::sycl::queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      void *mem = allocate_device(stream, n * sizeof(element_type));
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique_uninitialized(
        ::sycl::queue stream) {
      void *mem = allocate_device(stream, sizeof(T));
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique_uninitialized(
        size_t n, ::sycl::queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      void *mem = allocate_device(stream, n * sizeof(element_type));
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique_uninitialized(Args &&...) =
        delete;
  }  // namespace sycl
}  // namespace cms

#endif
