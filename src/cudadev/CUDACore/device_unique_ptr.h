#ifndef HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h

#include <functional>
#include <memory>

#include <cuda_runtime.h>

#include "CUDACore/allocate_device.h"
#include "CUDACore/currentDevice.h"

namespace cms {
  namespace cuda {
    namespace device {
      namespace impl {
        // Additional layer of types to distinguish from host::unique_ptr
        class DeviceDeleter {
        public:
          DeviceDeleter() = default;  // for edm::Wrapper
          DeviceDeleter(cudaStream_t stream) : stream_{stream} {}

          void operator()(void *ptr) {
            free_device(ptr, stream_);
          }

        private:
          cudaStream_t stream_ = cudaStreamDefault;
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;

      namespace impl {
        template <typename T>
        struct make_device_unique_selector {
          using non_array = cms::cuda::device::unique_ptr<T>;
        };
        template <typename T>
        struct make_device_unique_selector<T[]> {
          using unbounded_array = cms::cuda::device::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_device_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace device

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique(cudaStream_t stream) {
      static_assert(std::is_trivially_copyable<T>::value,
                    "Allocating with non-trivial copy on the device memory is not supported");
      void *mem = allocate_device(sizeof(T), stream);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(size_t n,
                                                                                              cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_copyable<element_type>::value,
                    "Allocating with non-trivial copy on the device memory is not supported");
      void *mem = allocate_device(n * sizeof(element_type), stream);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique(Args &&...) = delete;
  }  // namespace cuda
}  // namespace cms

#endif
