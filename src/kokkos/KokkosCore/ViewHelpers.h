#ifndef KokkosCore_ViewHelpers_h
#define KokkosCore_ViewHelpers_h

#include "KokkosCore/kokkosConfig.h"

namespace cms::kokkos {
  template <typename T, typename Space, typename... Args>
  auto make_view(char const* name, Args&&... args) {
    return Kokkos::View<T, Space>(Kokkos::ViewAllocateWithoutInitializing(name), std::forward<Args>(args)...);
  }

  template <typename T, typename... Properties>
  auto create_mirror_view(Kokkos::View<T, Properties...> const& src) {
    using Traits = Kokkos::ViewTraits<T, Properties...>;
    using HostSpace = typename MemSpaceTraits<typename Traits::memory_space>::HostSpace;
    return Kokkos::create_mirror_view(HostSpace(), src);
  }
}  // namespace cms::kokkos

#endif
