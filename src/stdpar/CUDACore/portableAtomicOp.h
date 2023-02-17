#ifndef HeterogeneousCore_CUDAUtilities_interface_portableAtomicOp_h
#define HeterogeneousCore_CUDAUtilities_interface_portableAtomicOp_h

#include <algorithm>
#include <atomic>
#include <concepts>
#include <type_traits>

#ifdef __NVCOMPILER
#include <nv/target>
#endif

namespace cms::cuda {

  template <std::totally_ordered T, typename Tp = std::add_pointer_t<T>>
  T atomicAdd(Tp address, T val) {
#ifdef __NVCOMPILER
    // We need to check if execution space is device or host, ::atomicAdd is undefined in host execution space
    if target (nv::target::is_device) {
      return ::atomicAdd(address, val);
    } else {
      std::atomic_ref<T> inc{*address};
      return inc.fetch_add(val);
    }
#else
    std::atomic_ref<T> inc{*address};
    return inc.fetch_add(val);
#endif
  }

  template <std::totally_ordered T, typename Tp = std::add_pointer_t<T>>
  T atomicMin(Tp address, T val) {
#ifdef __NVCOMPILER
    // We need to check if execution space is device or host, ::atomicMin is undefined in host execution space
    if target (nv::target::is_device) {
      return ::atomicMin(address, val);
    } else {
      std::atomic_ref<T> pa{*address};
      T old{pa.load()};
      while (!pa.compare_exchange_weak(old, std::min(old, val)))
        ;
      return old;
    }
#else
    std::atomic_ref<T> pa{*address};
    T old{pa.load()};
    while (!pa.compare_exchange_weak(old, std::min(old, val)))
      ;
    return old;
#endif
  }

  template <std::totally_ordered T, typename Tp = std::add_pointer_t<T>>
  T atomicMax(Tp address, T val) {
#ifdef __NVCOMPILER
    // We need to check if execution space is device or host, ::atomicMin is undefined in host execution space
    if target (nv::target::is_device) {
      return ::atomicMax(address, val);
    } else {
      std::atomic_ref<T> pa{*address};
      T old{pa.load()};
      while (!pa.compare_exchange_weak(old, std::max(old, val)))
        ;
      return old;
    }
#else
    std::atomic_ref<T> pa{*address};
    T old{pa.load()};
    while (!pa.compare_exchange_weak(old, std::max(old, val)))
      ;
    return old;
#endif
  }

  template <std::totally_ordered T, typename Tp = std::add_pointer_t<T>>
  T atomicInc(Tp address, T val) {
#ifdef __NVCOMPILER
    // We need to check if execution space is device or host, ::atomicInc is undefined in host execution space
    if target (nv::target::is_device) {
      return ::atomicInc(address, val);
    } else {
      std::atomic_ref<T> pa{*address};
      T old{pa.load()};
      while (!pa.compare_exchange_weak(old, ((old >= val) ? 0 : (old + 1))))
        ;
      return old;
    }
#else
    std::atomic_ref<T> pa{*address};
    T old{pa.load()};
    while (!pa.compare_exchange_weak(old, ((old >= val) ? 0 : (old + 1))))
      ;
    return old;
#endif
  }
}  // namespace cms::cuda

#endif