#ifndef KokkosCore_InitializeScopeGuard_h
#define KokkosCore_InitializeScopeGuard_h

#include "KokkosCore/kokkosConfigCommon.h"

#include <Kokkos_Core.hpp>

// Do not include this header in any executable that uses PluginManager, or any non-plugin library.
namespace kokkos_common {
  class InitializeScopeGuard : public InitializeScopeGuardBase {
  public:
    InitializeScopeGuard() {
      // for now pass in the default arguments
      Kokkos::InitArguments arguments;
      guard_ = std::make_unique<Kokkos::ScopeGuard>(arguments);
    }
    ~InitializeScopeGuard() override = default;

  private:
    std::unique_ptr<Kokkos::ScopeGuard> guard_;
  };
}  // namespace kokkos_common

#endif
