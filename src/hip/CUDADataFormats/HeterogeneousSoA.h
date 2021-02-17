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

  explicit HeterogeneousSoA(cms::hip::device::unique_ptr<T> &&p) : dm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(cms::hip::host::unique_ptr<T> &&p) : hm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(std::unique_ptr<T> &&p) : std_ptr(std::move(p)) {}

  auto const *get() const { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : std_ptr.get()); }

  auto const &operator*() const { return *get(); }

  auto const *operator->() const { return get(); }

  auto *get() { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : std_ptr.get()); }

  auto &operator*() { return *get(); }

  auto *operator->() { return get(); }

  // in reality valid only for GPU version...
  cms::hip::host::unique_ptr<T> toHostAsync(hipStream_t stream) const {
    assert(dm_ptr);
    auto ret = cms::hip::make_host_unique<T>(stream);
    cudaCheck(hipMemcpyAsync(ret.get(), dm_ptr.get(), sizeof(T), hipMemcpyDefault, stream));
    return ret;
  }

private:
  // a union wan't do it, a variant will not be more efficienct
  cms::hip::device::unique_ptr<T> dm_ptr;  //!
  cms::hip::host::unique_ptr<T> hm_ptr;    //!
  std::unique_ptr<T> std_ptr;               //!
};

namespace cms {
  namespace hipcompat {

    struct GPUTraits {
      template <typename T>
      using unique_ptr = cms::hip::device::unique_ptr<T>;

      template <typename T>
      static auto make_unique(hipStream_t stream) {
        return cms::hip::make_device_unique<T>(stream);
      }

      template <typename T>
      static auto make_unique(size_t size, hipStream_t stream) {
        return cms::hip::make_device_unique<T>(size, stream);
      }

      template <typename T>
      static auto make_host_unique(hipStream_t stream) {
        return cms::hip::make_host_unique<T>(stream);
      }

      template <typename T>
      static auto make_device_unique(hipStream_t stream) {
        return cms::hip::make_device_unique<T>(stream);
      }

      template <typename T>
      static auto make_device_unique(size_t size, hipStream_t stream) {
        return cms::hip::make_device_unique<T>(size, stream);
      }
    };

    struct HostTraits {
      template <typename T>
      using unique_ptr = cms::hip::host::unique_ptr<T>;

      template <typename T>
      static auto make_unique(hipStream_t stream) {
        return cms::hip::make_host_unique<T>(stream);
      }

      template <typename T>
      static auto make_host_unique(hipStream_t stream) {
        return cms::hip::make_host_unique<T>(stream);
      }

      template <typename T>
      static auto make_device_unique(hipStream_t stream) {
        return cms::hip::make_device_unique<T>(stream);
      }

      template <typename T>
      static auto make_device_unique(size_t size, hipStream_t stream) {
        return cms::hip::make_device_unique<T>(size, stream);
      }
    };

    struct CPUTraits {
      template <typename T>
      using unique_ptr = std::unique_ptr<T>;

      template <typename T>
      static auto make_unique(hipStream_t) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_unique(size_t size, hipStream_t) {
        return std::make_unique<T>(size);
      }

      template <typename T>
      static auto make_host_unique(hipStream_t) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_device_unique(hipStream_t) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_device_unique(size_t size, hipStream_t) {
        return std::make_unique<T>(size);
      }
    };

  }  // namespace hipcompat
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
  explicit HeterogeneousSoAImpl(hipStream_t stream);

  T const *get() const { return m_ptr.get(); }

  T *get() { return m_ptr.get(); }

  cms::hip::host::unique_ptr<T> toHostAsync(hipStream_t stream) const;

private:
  unique_ptr<T> m_ptr;  //!
};

template <typename T, typename Traits>
HeterogeneousSoAImpl<T, Traits>::HeterogeneousSoAImpl(hipStream_t stream) {
  m_ptr = Traits::template make_unique<T>(stream);
}

// in reality valid only for GPU version...
template <typename T, typename Traits>
cms::hip::host::unique_ptr<T> HeterogeneousSoAImpl<T, Traits>::toHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<T>(stream);
  cudaCheck(hipMemcpyAsync(ret.get(), get(), sizeof(T), hipMemcpyDefault, stream));
  return ret;
}

template <typename T>
using HeterogeneousSoAGPU = HeterogeneousSoAImpl<T, cms::hipcompat::GPUTraits>;
template <typename T>
using HeterogeneousSoACPU = HeterogeneousSoAImpl<T, cms::hipcompat::CPUTraits>;
template <typename T>
using HeterogeneousSoAHost = HeterogeneousSoAImpl<T, cms::hipcompat::HostTraits>;

#endif
