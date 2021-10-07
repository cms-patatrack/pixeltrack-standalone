#ifndef CUDADataFormatsCommonHeterogeneousSoA_H
#define CUDADataFormatsCommonHeterogeneousSoA_H

#include <cassert>

// a heterogeneous unique pointer...
template <typename T>
class HeterogeneousSoA {
public:
  using Product = T;

  HeterogeneousSoA() = default;  // make root happy
  ~HeterogeneousSoA() = default;
  HeterogeneousSoA(HeterogeneousSoA &&) = default;
  HeterogeneousSoA &operator=(HeterogeneousSoA &&) = default;

  explicit HeterogeneousSoA(std::unique_ptr<T> &&p) : std_ptr(std::move(p)) {}

  auto const *get() const { return std_ptr.get(); }

  auto const &operator*() const { return *get(); }

  auto const *operator->() const { return get(); }

  auto *get() { return std_ptr.get(); }

  auto &operator*() { return *get(); }

  auto *operator->() { return get(); }

private:
  std::unique_ptr<T> std_ptr;               //!
};

namespace cms {
  namespace cudacompat {

    struct CPUTraits {
      template <typename T>
      using unique_ptr = std::unique_ptr<T>;

      template <typename T>
      static auto make_unique(cudaStream_t) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_unique(size_t size, cudaStream_t) {
        return std::make_unique<T>(size);
      }

      template <typename T>
      static auto make_host_unique(cudaStream_t) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_device_unique(cudaStream_t) {
        return std::make_unique<T>();
      }

      template <typename T>
      static auto make_device_unique(size_t size, cudaStream_t) {
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
  explicit HeterogeneousSoAImpl(cudaStream_t stream);

  T const *get() const { return m_ptr.get(); }

  T *get() { return m_ptr.get(); }

private:
  unique_ptr<T> m_ptr;  //!
};

template <typename T, typename Traits>
HeterogeneousSoAImpl<T, Traits>::HeterogeneousSoAImpl(cudaStream_t stream) {
  m_ptr = Traits::template make_unique<T>(stream);
}

template <typename T>
using HeterogeneousSoACPU = HeterogeneousSoAImpl<T, cms::cudacompat::CPUTraits>;

#endif
