#ifndef KokkosCore_hintLightWeight_h
#define KokkosCore_hintLightWeight_h

#include <Kokkos_Core.hpp>

// shorthand because this will be used a lot
template <typename T>
auto hintLightWeight(T&& policy) {
  return Kokkos::Experimental::require(std::forward<T>(policy),
                                       Kokkos::Experimental::WorkItemProperty::HintLightWeight);
}

#endif  // KokkosCore_hintLightWeight_h
