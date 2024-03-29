#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/ExecSpaceCache.h"

#include <Kokkos_Core.hpp>


namespace kokkos_common {
  class InitializeScopeGuard::Impl {
  public:
    explicit Impl(std::vector<Backend> const& backends, Kokkos::InitArguments const& args) {
      Kokkos::Impl::pre_initialize(args);
      // Initialize SERIAL always
      Kokkos::Serial::impl_initialize();
      // Initialize THREADS always if enabled in Kokkos build
      // Not initializing tends to lead to "use of uninitialized execution space" errors at run time
#ifdef KOKKOS_ENABLE_THREADS
      Kokkos::Threads::impl_initialize(args.num_threads);
#endif
      if (std::find(backends.begin(), backends.end(), Backend::CUDA) != backends.end()) {
#ifdef KOKKOS_ENABLE_CUDA
        Kokkos::Cuda::impl_initialize();
#else
        throw std::runtime_error("CUDA backend was disabled at build time");
#endif
      }
      if (std::find(backends.begin(), backends.end(), Backend::HIP) != backends.end()) {
#ifdef KOKKOS_ENABLE_HIP
        Kokkos::Experimental::HIP::impl_initialize();
#else
        throw std::runtime_error("HIP backend was disabled at build time");
#endif
      }
      Kokkos::Impl::post_initialize(args);
    }

    ~Impl() {
#ifdef KOKKOS_ENABLE_CUDA
      cms::kokkos::getExecSpaceCache<Kokkos::Cuda>().clear();
#endif
#ifdef KOKKOS_ENABLE_HIP
      cms::kokkos::getExecSpaceCache<Kokkos::Experimental::HIP>().clear();
#endif
      Kokkos::finalize();
    }
  };

  InitializeScopeGuard::InitializeScopeGuard(std::vector<Backend> const& backends, int numberOfInnerThreads) {
    // for now pass in the default arguments
    Kokkos::InitArguments arguments(numberOfInnerThreads);
    auto env_tool_lib = std::getenv("KOKKOS_PROFILE_LIBRARY");
    if (env_tool_lib != nullptr) {
      arguments.tool_lib = env_tool_lib;
    }
    pimpl_ = std::make_unique<Impl>(backends, arguments);
  }

  InitializeScopeGuard::~InitializeScopeGuard() = default;
}  // namespace kokkos_common
