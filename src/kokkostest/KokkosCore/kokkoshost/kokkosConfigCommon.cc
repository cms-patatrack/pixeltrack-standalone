#include "KokkosCore/kokkosConfigCommon.h"

#include <Kokkos_Core.hpp>

namespace kokkos_common {
  class InitializeScopeGuard::Impl {
  public:
    explicit Impl(const Kokkos::InitArguments& args) : guard(args) {}

  private:
    Kokkos::ScopeGuard guard;
  };

  InitializeScopeGuard::InitializeScopeGuard() {
    // for now pass in the default arguments
    Kokkos::InitArguments arguments;
    pimpl_ = std::make_unique<Impl>(arguments);
  }

  InitializeScopeGuard::~InitializeScopeGuard() = default;
}  // namespace kokkos_common
