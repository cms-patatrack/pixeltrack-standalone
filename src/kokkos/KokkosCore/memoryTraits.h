#ifndef KokkosCore_memoryTraits_h
#define KokkosCore_memoryTraits_h

#include <Kokkos_Core.hpp>

// shorthand because this will be used a lot
using Restrict = Kokkos::MemoryTraits<Kokkos::Restrict>;
using RestrictUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>;

namespace cms::kokkos {
  template <typename T, typename... Args>
  auto make_const(Kokkos::View<T, Args...> const& view) {
    return Kokkos::View<const T, Args...>(view);
  }
}  // namespace cms::kokkos

#endif
