#ifndef KokkosCore_atomic_h
#define KokkosCore_atomic_h

#include <Kokkos_Core.hpp>

// Abstractions to optionally disable atomicity of atomics when using serial backend only
// TODO: It would be better to do this only depending on the "active
// backend", regardless of how the Kokkos runtime library has been
// configured. I can see two options
// - Template over the execution space. But execution space objects are not available in the device code.
// - Enclose everything in KOKKOS_NAMESPACE namespace for the "poor-man" templating we're doing with that
// To be looked into after performance impact is demonstrated with Serial-only build
namespace cms::kokkos {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION T atomic_fetch_add(T* dest, const T val) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
// some sanity checks
#ifndef KOKKOS_BACKEND_SERIAL
#error "KOKKOS_SERIALONLY_DISABLE_ATOMICS can be used only with a build that has only the Serial backend enabled"
#endif
    auto old = *dest;
    *dest += val;
    return old;
#else
    return Kokkos::atomic_fetch_add(dest, val);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION void atomic_add(T* dest, const T val) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
    *dest += val;
#else
    return Kokkos::atomic_add(dest, val);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION T atomic_fetch_sub(T* dest, const T val) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
    auto old = *dest;
    *dest -= val;
    return old;
#else
    return Kokkos::atomic_fetch_sub(dest, val);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION void atomic_sub(T* dest, const T val) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
    *dest -= val;
#else
    return Kokkos::atomic_sub(dest, val);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION void atomic_decrement(T* dest) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
    --(*dest);
#else
    Kokkos::atomic_decrement(dest);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION T atomic_fetch_min(T* dest, const T val) {
#if defined(KOKKOS_BACKEND_SERIAL) && !defined(KOKKOS_SERIAL_ENABLE_ATOMICS)
    auto old = *dest;
    *dest = std::min(*dest, val);
    return old;
#else
    return Kokkos::atomic_fetch_min(dest, val);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION T atomic_min_fetch(T* dest, const T val) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
    auto ret = std::min(*dest, val);
    *dest = ret;
    return ret;
#else
    return Kokkos::atomic_min_fetch(dest, val);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION T atomic_max_fetch(T* dest, const T val) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
    auto ret = std::max(*dest, val);
    *dest = ret;
    return ret;
#else
    return Kokkos::atomic_max_fetch(dest, val);
#endif
  }

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION T atomic_compare_exchange(T* dest, const T compare, const T val) {
#ifdef KOKKOS_SERIALONLY_DISABLE_ATOMICS
    auto old = *dest;
    *dest = (old == compare) ? val : old;
    return old;
#else
    return Kokkos::atomic_compare_exchange(dest, compare, val);
#endif
  }
}  // namespace cms::kokkos

#endif
