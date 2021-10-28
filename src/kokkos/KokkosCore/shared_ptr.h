#ifndef HeterogeneousCore_KokkosUtilities_interface_shared_ptr_h
#define HeterogeneousCore_KokkosUtilities_interface_shared_ptr_h

#include <memory>

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/memoryTraits.h"

namespace cms::kokkos {
  namespace impl {
    template <typename MemSpace>
    class Deleter {
    public:
      void operator()(void* ptr) const { Kokkos::kokkos_free<MemSpace>(ptr); }
    };
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

  template <typename T, typename MemSpace>
  std::enable_if_t<!std::is_array_v<T>, shared_ptr<T, MemSpace>> make_shared() {
    void* mem = Kokkos::kokkos_malloc<MemSpace>(sizeof(T));
    return shared_ptr<T, MemSpace>(static_cast<T*>(mem), impl::Deleter<MemSpace>());
  }

  template <typename T, typename MemSpace>
  std::enable_if_t<std::is_array_v<T>, shared_ptr<T, MemSpace>> make_shared(size_t n) {
    using element_type = typename std::remove_extent<T>::type;
    void* mem = Kokkos::kokkos_malloc<MemSpace>(sizeof(element_type) * n);
    return shared_ptr<T, MemSpace>(static_cast<element_type*>(mem), n, impl::Deleter<MemSpace>());
  }

  template <typename T, typename MemSpace>
  auto make_mirror_shared(shared_ptr<T, MemSpace> const& src) {
    using MirrorSpace = typename MemSpaceTraits<MemSpace>::HostSpace;
    if constexpr (std::is_same_v<MemSpace, MirrorSpace>) {
      return src;
    } else if constexpr (std::is_array_v<T>) {
      return make_shared<T, MirrorSpace>(src.size());
    } else {
      return make_shared<T, MirrorSpace>();
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
