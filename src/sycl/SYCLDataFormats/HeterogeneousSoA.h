#ifndef SYCLDataFormatsCommonHeterogeneousSoA_H
#define SYCLDataFormatsCommonHeterogeneousSoA_H

#include <cassert>

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

// a heterogeneous unique pointer...
template <typename T>
class HeterogeneousSoA {
public:
  using Product = T;

  HeterogeneousSoA() = default;  // make root happy
  ~HeterogeneousSoA() = default;
  HeterogeneousSoA(HeterogeneousSoA &&) = default;
  HeterogeneousSoA &operator=(HeterogeneousSoA &&) = default;

  explicit HeterogeneousSoA(cms::sycltools::device::unique_ptr<T> &&p) : dm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(cms::sycltools::host::unique_ptr<T> &&p) : hm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(std::unique_ptr<T> &&p) : std_ptr(std::move(p)) {}

  auto const *get() const { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : std_ptr.get()); }

  auto const &operator*() const { return *get(); }

  auto const *operator->() const { return get(); }

  auto *get() { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : std_ptr.get()); }

  auto &operator*() { return *get(); }

  auto *operator->() { return get(); }

  // in reality valid only for GPU version...
  cms::sycltools::host::unique_ptr<T> toHostAsync(sycl::queue stream) const {
    assert(dm_ptr);
    auto ret = cms::sycltools::make_host_unique<T>(stream);
    stream.memcpy(ret.get(), dm_ptr.get(), sizeof(T));
    return ret;
  }

private:
  // a union wan't do it, a variant will not be more efficienct
  cms::sycltools::device::unique_ptr<T> dm_ptr;  //!
  cms::sycltools::host::unique_ptr<T> hm_ptr;    //!
  std::unique_ptr<T> std_ptr;                    //!
};

#endif
