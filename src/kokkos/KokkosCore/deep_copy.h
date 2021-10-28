#ifndef HeterogeneousCore_KokkosUtilities_interface_deep_copy_h
#define HeterogeneousCore_KokkosUtilities_interface_deep_copy_h

#include "KokkosCore/shared_ptr.h"

namespace cms::kokkos {
  template <typename ExecSpace, typename T, typename MemSpaceDst, typename MemSpaceSrc>
  void deep_copy(ExecSpace const& execSpace, shared_ptr<T, MemSpaceDst>& dst, shared_ptr<T, MemSpaceSrc> const& src) {
    if constexpr (std::is_same_v<MemSpaceDst, MemSpaceSrc>) {
      if (dst.get() == src.get()) {
        return;
      }
    }
    auto v_dst = to_view(dst);
    auto v_src = to_view(src);
    Kokkos::deep_copy(execSpace, v_dst, v_src);
  }
}  // namespace cms::kokkos

#endif
