#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/memoryTraits.h"

template <typename MemorySpace>
class SiPixelClustersKokkos {
public:
  SiPixelClustersKokkos() = default;
  explicit SiPixelClustersKokkos(size_t maxClusters)
      : moduleStart_d{Kokkos::ViewAllocateWithoutInitializing("moduleStart_d"), maxClusters + 1},
        clusInModule_d{Kokkos::ViewAllocateWithoutInitializing("clusInModule_d"), maxClusters},
        moduleId_d{Kokkos::ViewAllocateWithoutInitializing("moduleId_d"), maxClusters},
        clusModuleStart_d{Kokkos::ViewAllocateWithoutInitializing("clusModuleStart_d"), maxClusters + 1} {}
  ~SiPixelClustersKokkos() = default;

  SiPixelClustersKokkos(const SiPixelClustersKokkos &) = delete;
  SiPixelClustersKokkos &operator=(const SiPixelClustersKokkos &) = delete;
  SiPixelClustersKokkos(SiPixelClustersKokkos &&) = default;
  SiPixelClustersKokkos &operator=(SiPixelClustersKokkos &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

  Kokkos::View<uint32_t *, MemorySpace> moduleStart() { return moduleStart_d; }
  Kokkos::View<uint32_t *, MemorySpace> clusInModule() { return clusInModule_d; }
  Kokkos::View<uint32_t *, MemorySpace> moduleId() { return moduleId_d; }
  Kokkos::View<uint32_t *, MemorySpace> clusModuleStart() { return clusModuleStart_d; }

  Kokkos::View<uint32_t const *, MemorySpace> moduleStart() const { return moduleStart_d; }
  Kokkos::View<uint32_t const *, MemorySpace> clusInModule() const { return clusInModule_d; }
  Kokkos::View<uint32_t const *, MemorySpace> moduleId() const { return moduleId_d; }
  Kokkos::View<uint32_t const *, MemorySpace> clusModuleStart() const { return clusModuleStart_d; }

  Kokkos::View<uint32_t const *, MemorySpace> c_moduleStart() const { return moduleStart_d; }
  Kokkos::View<uint32_t const *, MemorySpace> c_clusInModule() const { return clusInModule_d; }
  Kokkos::View<uint32_t const *, MemorySpace> c_moduleId() const { return moduleId_d; }
  Kokkos::View<uint32_t const *, MemorySpace> c_clusModuleStart() const { return clusModuleStart_d; }

  class DeviceConstView {
  public:
    // DeviceConstView() = default;

    KOKKOS_FORCEINLINE_FUNCTION uint32_t moduleStart(int i) const { return moduleStart_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint32_t clusInModule(int i) const { return clusInModule_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint32_t moduleId(int i) const { return moduleId_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint32_t clusModuleStart(int i) const { return clusModuleStart_[i]; }

    friend SiPixelClustersKokkos;

    // private:
    Kokkos::View<uint32_t const *, MemorySpace, Restrict> moduleStart_;
    Kokkos::View<uint32_t const *, MemorySpace, Restrict> clusInModule_;
    Kokkos::View<uint32_t const *, MemorySpace, Restrict> moduleId_;
    Kokkos::View<uint32_t const *, MemorySpace, Restrict> clusModuleStart_;
  };

  DeviceConstView view() const { return DeviceConstView{moduleStart_d, clusInModule_d, moduleId_d, clusModuleStart_d}; }

private:
  Kokkos::View<uint32_t *, MemorySpace> moduleStart_d;   // index of the first pixel of each module
  Kokkos::View<uint32_t *, MemorySpace> clusInModule_d;  // number of clusters found in each module
  Kokkos::View<uint32_t *, MemorySpace> moduleId_d;      // module id of each module

  // originally from rechits
  Kokkos::View<uint32_t *, MemorySpace> clusModuleStart_d;  // index of the first cluster of each module

#ifdef TODO
  cms::cuda::device::unique_ptr<DeviceConstView> view_d;  // "me" pointer
#endif

  uint32_t nClusters_h;
};

#endif
