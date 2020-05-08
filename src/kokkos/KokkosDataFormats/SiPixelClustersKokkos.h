#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "KokkosCore/kokkosConfig.h"

template <typename MemorySpace>
class SiPixelClustersKokkos {
public:
  SiPixelClustersKokkos() = default;
  explicit SiPixelClustersKokkos(size_t maxClusters)
      : moduleStart_d{"moduleStart_d", maxClusters + 1},
        clusInModule_d{"clusInModule_d", maxClusters},
        moduleId_d{"moduleId_d", maxClusters},
        clusModuleStart_d{"clusModuleStart_d", maxClusters} {}
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

#ifdef TODO
  class DeviceConstView {
  public:
    // DeviceConstView() = default;

    __device__ __forceinline__ uint32_t moduleStart(int i) const { return __ldg(moduleStart_ + i); }
    __device__ __forceinline__ uint32_t clusInModule(int i) const { return __ldg(clusInModule_ + i); }
    __device__ __forceinline__ uint32_t moduleId(int i) const { return __ldg(moduleId_ + i); }
    __device__ __forceinline__ uint32_t clusModuleStart(int i) const { return __ldg(clusModuleStart_ + i); }

    friend SiPixelClustersKokkos;

    //   private:
    uint32_t const *moduleStart_;
    uint32_t const *clusInModule_;
    uint32_t const *moduleId_;
    uint32_t const *clusModuleStart_;
  };

  DeviceConstView *view() const { return view_d.get(); }
#endif

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
