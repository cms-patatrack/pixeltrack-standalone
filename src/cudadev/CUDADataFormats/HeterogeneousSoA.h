#ifndef CUDADataFormatsCommonHeterogeneousSoA_H
#define CUDADataFormatsCommonHeterogeneousSoA_H

#include <cassert>

#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

// a heterogeneous unique pointer...
template <typename T>
class HeterogeneousSoA {
public:
  using Product = T;

  HeterogeneousSoA() = default;  // make root happy
  ~HeterogeneousSoA() = default;
  HeterogeneousSoA(HeterogeneousSoA &&) = default;
  HeterogeneousSoA &operator=(HeterogeneousSoA &&) = default;

  explicit HeterogeneousSoA(cms::cuda::device::unique_ptr<T> &&p) : dm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(cms::cuda::host::unique_ptr<T> &&p) : hm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(std::unique_ptr<T> &&p) : std_ptr(std::move(p)) {}

  auto const *get() const { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : std_ptr.get()); }

  auto const &operator*() const { return *get(); }

  auto const *operator->() const { return get(); }

  auto *get() { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : std_ptr.get()); }

  auto &operator*() { return *get(); }

  auto *operator->() { return get(); }

  // in reality valid only for GPU version...
  cms::cuda::host::unique_ptr<T> toHostAsync(cms::cuda::Context const &ctx) const {
    assert(dm_ptr);
    auto ret = cms::cuda::make_host_unique<T>(ctx);
    cudaCheck(cudaMemcpyAsync(ret.get(), dm_ptr.get(), sizeof(T), cudaMemcpyDefault, ctx.stream()));
    return ret;
  }

private:
  // a union wan't do it, a variant will not be more efficienct
  cms::cuda::device::unique_ptr<T> dm_ptr;  //!
  cms::cuda::host::unique_ptr<T> hm_ptr;    //!
  std::unique_ptr<T> std_ptr;               //!
};

namespace cms {
  namespace cudacompat {

    struct GPUTraits {
      template <typename T>
      using unique_ptr = cms::cuda::device::unique_ptr<T>;

      template <typename T>
      static auto make_unique(cms::cuda::DeviceAllocatorContext const &ctx) {
        return cms::cuda::make_device_unique<T>(ctx);
      }

      template <typename T>
      static auto make_unique(size_t size, cms::cuda::DeviceAllocatorContext const &ctx) {
        return cms::cuda::make_device_unique<T>(size, ctx);
      }

      template <typename T>
      static auto make_host_unique(cms::cuda::HostAllocatorContext const &ctx) {
        return cms::cuda::make_host_unique<T>(ctx);
      }

      template <typename T>
      static auto make_device_unique(cms::cuda::DeviceAllocatorContext const &ctx) {
        return cms::cuda::make_device_unique<T>(ctx);
      }

      template <typename T>
      static auto make_device_unique(size_t size, cms::cuda::DeviceAllocatorContext const &ctx) {
        return cms::cuda::make_device_unique<T>(size, ctx);
      }
    };

    struct HostTraits {
      template <typename T>
      using unique_ptr = cms::cuda::host::unique_ptr<T>;

      template <typename T>
      static auto make_unique(cms::cuda::HostAllocatorContext const &ctx) {
        return cms::cuda::make_host_unique<T>(ctx);
      }

      template <typename T>
      static auto make_host_unique(cms::cuda::HostAllocatorContext const &ctx) {
        return cms::cuda::make_host_unique<T>(ctx);
      }

      template <typename T>
      static auto make_device_unique(cms::cuda::DeviceAllocatorContext const &ctx) {
        return cms::cuda::make_device_unique<T>(ctx);
      }

      template <typename T>
      static auto make_device_unique(size_t size, cms::cuda::DeviceAllocatorContext const &ctx) {
        return cms::cuda::make_device_unique<T>(size, ctx);
      }
    };

    struct CPUTraits {
      template <typename T>
      using unique_ptr = std::unique_ptr<T>;

      template <typename T>
      static auto make_unique(cms::cuda::DeviceAllocatorContext const &) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_unique(size_t size, cms::cuda::DeviceAllocatorContext const &) {
        return std::make_unique<T>(size);
      }

      template <typename T>
      static auto make_host_unique(cms::cuda::HostAllocatorContext const &) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_device_unique(cms::cuda::DeviceAllocatorContext const &) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_device_unique(size_t size, cms::cuda::DeviceAllocatorContext const &) {
        return std::make_unique<T>(size);
      }
    };

  }  // namespace cudacompat
}  // namespace cms

// a heterogeneous unique pointer (of a different sort) ...
template <typename T, typename Traits>
class HeterogeneousSoAImpl {
public:
  template <typename V>
  using unique_ptr = typename Traits::template unique_ptr<V>;

  HeterogeneousSoAImpl() = default;  // make root happy
  ~HeterogeneousSoAImpl() = default;
  HeterogeneousSoAImpl(HeterogeneousSoAImpl &&) = default;
  HeterogeneousSoAImpl &operator=(HeterogeneousSoAImpl &&) = default;

  explicit HeterogeneousSoAImpl(unique_ptr<T> &&p) : m_ptr(std::move(p)) {}
  explicit HeterogeneousSoAImpl(cms::cuda::DeviceAllocatorContext const &ctx);

  T const *get() const { return m_ptr.get(); }

  T *get() { return m_ptr.get(); }

  cms::cuda::host::unique_ptr<T> toHostAsync(cms::cuda::Context const &ctx) const;

private:
  unique_ptr<T> m_ptr;  //!
};

template <typename T, typename Traits>
HeterogeneousSoAImpl<T, Traits>::HeterogeneousSoAImpl(cms::cuda::DeviceAllocatorContext const &ctx) {
  m_ptr = Traits::template make_unique<T>(ctx);
}

// in reality valid only for GPU version...
template <typename T, typename Traits>
cms::cuda::host::unique_ptr<T> HeterogeneousSoAImpl<T, Traits>::toHostAsync(cms::cuda::Context const &ctx) const {
  auto ret = cms::cuda::make_host_unique<T>(ctx);
  cudaCheck(cudaMemcpyAsync(ret.get(), get(), sizeof(T), cudaMemcpyDefault, ctx.stream()));
  return ret;
}

template <typename T>
using HeterogeneousSoAGPU = HeterogeneousSoAImpl<T, cms::cudacompat::GPUTraits>;
template <typename T>
using HeterogeneousSoACPU = HeterogeneousSoAImpl<T, cms::cudacompat::CPUTraits>;
template <typename T>
using HeterogeneousSoAHost = HeterogeneousSoAImpl<T, cms::cudacompat::HostTraits>;

#endif
