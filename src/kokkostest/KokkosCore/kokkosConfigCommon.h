#ifndef KokkosCore_kokkosConfigCommon_h
#define KokkosCore_kokkosConfigCommon_h

#include <vector>
#include <memory>

// This header needs to be #included in a file that may not be
// compiled with nvcc.
namespace kokkos_common {
  class InitializeScopeGuard {
  public:
    enum class Backend { SERIAL, PTHREAD, CUDA, HIP };

    explicit InitializeScopeGuard(std::vector<Backend> const& backends, int numberOfInnerThreads = 1);
    ~InitializeScopeGuard();

  private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
  };
}  // namespace kokkos_common

#endif
