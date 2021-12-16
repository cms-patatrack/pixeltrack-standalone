#ifndef HeterogeneousCore_KokkosUtilities_interface_shared_ptr_h
#define HeterogeneousCore_KokkosUtilities_interface_shared_ptr_h

#include <memory>

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/memoryTraits.h"

#ifdef KOKKOS_ENABLE_CUDA
#include "CUDACore/allocate_device.h"
#include "CUDACore/allocate_host.h"
#endif

namespace cms::kokkos {
  namespace impl {
    template <typename MemSpace>
    struct Deleter {
      void operator()(void* ptr) const { Kokkos::kokkos_free<MemSpace>(ptr); }
    };

    template <typename MemSpace>
    struct Allocate {
      template <typename ExecSpace>
      static void* allocate(size_t bytes, ExecSpace const& execSpace) {
        return Kokkos::kokkos_malloc<MemSpace>(bytes);
      }

      template <typename ExecSpace>
      static Deleter<MemSpace> make_deleter(ExecSpace const& execSpace) {
        return Deleter<MemSpace>();
      }
    };

#ifdef KOKKOS_ENABLE_CUDA
    template <>
    struct Deleter<Kokkos::CudaSpace> {
      Deleter(cudaStream_t stream) : stream_{stream} {}

      void operator()(void* ptr) const { cms::cuda::free_device(ptr, stream_); }

    private:
      cudaStream_t stream_ = cudaStreamDefault;
    };
    template <>
    struct Deleter<Kokkos::CudaHostPinnedSpace> {
      void operator()(void* ptr) const { cms::cuda::free_host(ptr); }
    };

    template <>
    struct Allocate<Kokkos::CudaSpace> {
      static void* allocate(size_t bytes, Kokkos::Cuda const& execSpace) {
        return cms::cuda::allocate_device(execSpace.cuda_device(), bytes, execSpace.cuda_stream());
      }

      static Deleter<Kokkos::CudaSpace> make_deleter(Kokkos::Cuda const& execSpace) {
        return Deleter<Kokkos::CudaSpace>(execSpace.cuda_stream());
      }
    };
    template <>
    struct Allocate<Kokkos::CudaHostPinnedSpace> {
      static void* allocate(size_t bytes, Kokkos::Cuda const& execSpace) {
        return cms::cuda::allocate_host(execSpace.cuda_device(), bytes, execSpace.cuda_stream());
      }

      static Deleter<Kokkos::CudaHostPinnedSpace> make_deleter(Kokkos::Cuda const& execSpace) {
        return Deleter<Kokkos::CudaHostPinnedSpace>();
      }
    };

#endif
  }  // namespace impl

  template <typename T, typename MemorySpace>
  class shared_ptr {
  public:
    shared_ptr() = default;
    template <typename Deleter>
    shared_ptr(T* ptr, Deleter&& d) : ptr_(ptr, std::forward<Deleter>(d)) {}

    T* get() { return ptr_.get(); }
    T const* get() const { return ptr_.get(); }

    T& operator*() { return *ptr_; }
    T const& operator*() const { return *ptr_; }

    T* operator->() { return ptr_.get(); }
    T const* operator->() const { return ptr_.get(); }

    void reset() { ptr_.reset(); }

  private:
    std::shared_ptr<T> ptr_;
  };

  template <typename T, typename MemorySpace>
  class shared_ptr<T[], MemorySpace> {
  public:
    shared_ptr() = default;
    template <typename Deleter>
    shared_ptr(T* ptr, size_t s, Deleter d) : ptr_(ptr, d), size_(s) {}

    T* get() { return ptr_.get(); }
    T const* get() const { return ptr_.get(); }

    size_t size() const { return size_; }

    T& operator[](std::ptrdiff_t i) { return ptr_[i]; }
    T const& operator[](std::ptrdiff_t i) const { return ptr_[i]; }

    void reset() { ptr_.reset(); }

  private:
    std::shared_ptr<T[]> ptr_;
    size_t size_ = 0;
  };

  template <typename T, typename MemSpace, typename ExecSpace>
  std::enable_if_t<!std::is_array_v<T>, shared_ptr<T, MemSpace>> make_shared(ExecSpace const& execSpace) {
    using Allocator = impl::Allocate<MemSpace>;
    void* mem = Allocator::allocate(sizeof(T), execSpace);
    return shared_ptr<T, MemSpace>(static_cast<T*>(mem), Allocator::make_deleter(execSpace));
  }

  template <typename T, typename MemSpace, typename ExecSpace>
  std::enable_if_t<std::is_array_v<T>, shared_ptr<T, MemSpace>> make_shared(size_t n, ExecSpace const& execSpace) {
    using Allocator = impl::Allocate<MemSpace>;
    using element_type = typename std::remove_extent<T>::type;
    void* mem = Allocator::allocate(sizeof(element_type) * n, execSpace);
    return shared_ptr<T, MemSpace>(static_cast<element_type*>(mem), n, Allocator::make_deleter(execSpace));
  }

  template <typename T, typename MemSpace, typename ExecSpace>
  auto make_mirror_shared(shared_ptr<T, MemSpace> const& src, ExecSpace const& execSpace) {
    using MirrorSpace = typename MemSpaceTraits<MemSpace>::HostSpace;
    if constexpr (std::is_same_v<MemSpace, MirrorSpace>) {
      return src;
    } else if constexpr (std::is_array_v<T>) {
      return make_shared<T, MirrorSpace>(src.size(), execSpace);
    } else {
      return make_shared<T, MirrorSpace>(execSpace);
    }
    return shared_ptr<T, MirrorSpace>();
  }

  template <typename T, typename MemSpace>
  auto to_view(shared_ptr<T, MemSpace>& ptr) {
    return Kokkos::View<T, MemSpace, RestrictUnmanaged>(ptr.get());
  }

  template <typename T, typename MemSpace>
  auto to_view(shared_ptr<T, MemSpace> const& ptr) {
    return Kokkos::View<T const, MemSpace, RestrictUnmanaged>(ptr.get());
  }

  template <typename T, typename MemSpace>
  auto to_view(shared_ptr<T[], MemSpace>& ptr) {
    return Kokkos::View<T*, MemSpace, RestrictUnmanaged>(ptr.get(), ptr.size());
  }

  template <typename T, typename MemSpace>
  auto to_view(shared_ptr<T[], MemSpace> const& ptr) {
    return Kokkos::View<T const*, MemSpace, RestrictUnmanaged>(ptr.get(), ptr.size());
  }
}  // namespace cms::kokkos

#endif
