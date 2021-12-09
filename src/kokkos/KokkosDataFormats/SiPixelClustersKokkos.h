#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/memoryTraits.h"
#include "KokkosCore/ViewHelpers.h"
#include "KokkosCore/deep_copy.h"
#include "KokkosCore/shared_ptr.h"

template <typename MemorySpace>
class SiPixelClustersKokkos {
public:
  template <typename T>
  using View = Kokkos::View<T, MemorySpace, RestrictUnmanaged>;

  SiPixelClustersKokkos() = default;
  template <typename ExecSpace>
  explicit SiPixelClustersKokkos(size_t maxClusters, ExecSpace const &execSpace)
      : moduleStart_d{cms::kokkos::make_shared<uint32_t[], MemorySpace>(maxClusters + 1, execSpace)},
        clusInModule_d{cms::kokkos::make_shared<uint32_t[], MemorySpace>(maxClusters, execSpace)},
        moduleId_d{cms::kokkos::make_shared<uint32_t[], MemorySpace>(maxClusters, execSpace)},
        clusModuleStart_d{cms::kokkos::make_shared<uint32_t[], MemorySpace>(maxClusters + 1, execSpace)} {}
  ~SiPixelClustersKokkos() = default;

  SiPixelClustersKokkos(const SiPixelClustersKokkos &) = delete;
  SiPixelClustersKokkos &operator=(const SiPixelClustersKokkos &) = delete;
  SiPixelClustersKokkos(SiPixelClustersKokkos &&) = default;
  SiPixelClustersKokkos &operator=(SiPixelClustersKokkos &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

  View<uint32_t *> moduleStart() { return cms::kokkos::to_view(moduleStart_d); }
  View<uint32_t *> clusInModule() { return cms::kokkos::to_view(clusInModule_d); }
  View<uint32_t *> moduleId() { return cms::kokkos::to_view(moduleId_d); }
  View<uint32_t *> clusModuleStart() { return cms::kokkos::to_view(clusModuleStart_d); }

  View<uint32_t const *> moduleStart() const { return cms::kokkos::to_view(moduleStart_d); }
  View<uint32_t const *> clusInModule() const { return cms::kokkos::to_view(clusInModule_d); }
  View<uint32_t const *> moduleId() const { return cms::kokkos::to_view(moduleId_d); }
  View<uint32_t const *> clusModuleStart() const { return cms::kokkos::to_view(clusModuleStart_d); }

  View<uint32_t const *> c_moduleStart() const { return cms::kokkos::to_view(moduleStart_d); }
  View<uint32_t const *> c_clusInModule() const { return cms::kokkos::to_view(clusInModule_d); }
  View<uint32_t const *> c_moduleId() const { return cms::kokkos::to_view(moduleId_d); }
  View<uint32_t const *> c_clusModuleStart() const { return cms::kokkos::to_view(clusModuleStart_d); }

  template <typename ExecSpace>
  auto clusInModuleToHostAsync(ExecSpace const &execSpace) const {
    auto host = cms::kokkos::make_mirror_shared(clusInModule_d, execSpace);
    cms::kokkos::deep_copy(execSpace, host, clusInModule_d);
    return host;
  }

  class DeviceConstView {
  public:
    // DeviceConstView() = default;

    KOKKOS_FORCEINLINE_FUNCTION uint32_t moduleStart(int i) const { return moduleStart_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint32_t clusInModule(int i) const { return clusInModule_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint32_t moduleId(int i) const { return moduleId_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint32_t clusModuleStart(int i) const { return clusModuleStart_[i]; }

    friend SiPixelClustersKokkos;

    // private:
    Kokkos::View<uint32_t const *, MemorySpace, RestrictUnmanaged> moduleStart_;
    Kokkos::View<uint32_t const *, MemorySpace, RestrictUnmanaged> clusInModule_;
    Kokkos::View<uint32_t const *, MemorySpace, RestrictUnmanaged> moduleId_;
    Kokkos::View<uint32_t const *, MemorySpace, RestrictUnmanaged> clusModuleStart_;
  };

  DeviceConstView view() const { return DeviceConstView{moduleStart(), clusInModule(), moduleId(), clusModuleStart()}; }

private:
  cms::kokkos::shared_ptr<uint32_t[], MemorySpace> moduleStart_d;   // index of the first pixel of each module
  cms::kokkos::shared_ptr<uint32_t[], MemorySpace> clusInModule_d;  // number of clusters found in each module
  cms::kokkos::shared_ptr<uint32_t[], MemorySpace> moduleId_d;      // module id of each module

  // originally from rechits
  cms::kokkos::shared_ptr<uint32_t[], MemorySpace> clusModuleStart_d;  // index of the first cluster of each module

#ifdef TODO
  cms::cuda::device::unique_ptr<DeviceConstView> view_d;  // "me" pointer
#endif

  uint32_t nClusters_h;
};

#endif
