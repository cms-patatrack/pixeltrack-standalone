#include "KokkosCore/kokkosConfigCommon.h"

#include <Kokkos_Core.hpp>

namespace kokkos_common {
  class InitializeScopeGuard::Impl {
  public:
    explicit Impl(std::vector<Backend> const& backends, Kokkos::InitArguments const& args) {
      Kokkos::Impl::pre_initialize(args);
      if (std::find(backends.begin(), backends.end(), Backend::SERIAL) != backends.end()) {
        Kokkos::Serial::impl_initialize();
      }
      if (std::find(backends.begin(), backends.end(), Backend::CUDA) != backends.end()) {
        // CUDA execution space requires a host execution space as well
        Kokkos::Serial::impl_initialize();
        Kokkos::Cuda::impl_initialize();
      }
      Kokkos::Impl::post_initialize(args);
    }

    ~Impl() { Kokkos::finalize(); }
  };

  InitializeScopeGuard::InitializeScopeGuard(std::vector<Backend> const& backends) {
    // for now pass in the default arguments
    Kokkos::InitArguments arguments;
    pimpl_ = std::make_unique<Impl>(backends, arguments);
  }

  InitializeScopeGuard::~InitializeScopeGuard() = default;
}  // namespace kokkos_common
