#ifndef HeterogeneousCore_CUDAUtilities_interface_portableAtomicOp_h
#define HeterogeneousCore_CUDAUtilities_interface_portableAtomicOp_h

#include <concepts>
#include <type_traits>

#ifdef __NVCOMPILER
#include <nv/target>
#endif

namespace cms::cuda {

  template<std::floating_point T, typename Tp=std::add_pointer_t<T>>
  T atomicAdd(Tp address, T val) {
#ifdef __NVCOMPILER
    // We need to check if execution space is device or host, ::atomicAdd is undefined in host execution space 
    if target(nv::target::is_device) {
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
}

#endif