#ifndef AlpakaDataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define AlpakaDataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "AlpakaCore/device_unique_ptr.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelClustersAlpaka {
  public:
    SiPixelClustersAlpaka() = default;
    explicit SiPixelClustersAlpaka(size_t maxClusters)
        : moduleStart_d{cms::alpakatools::make_device_unique<uint32_t>(maxClusters + 1)},
          clusInModule_d{cms::alpakatools::make_device_unique<uint32_t>(maxClusters)},
          moduleId_d{cms::alpakatools::make_device_unique<uint32_t>(maxClusters)},
          clusModuleStart_d{cms::alpakatools::make_device_unique<uint32_t>(maxClusters + 1)} {}
    ~SiPixelClustersAlpaka() = default;

    SiPixelClustersAlpaka(const SiPixelClustersAlpaka &) = delete;
    SiPixelClustersAlpaka &operator=(const SiPixelClustersAlpaka &) = delete;
    SiPixelClustersAlpaka(SiPixelClustersAlpaka &&) = default;
    SiPixelClustersAlpaka &operator=(SiPixelClustersAlpaka &&) = default;

    void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

    uint32_t nClusters() const { return nClusters_h; }

    uint32_t *moduleStart() { return moduleStart_d.get(); }
    uint32_t *clusInModule() { return clusInModule_d.get(); }
    uint32_t *moduleId() { return moduleId_d.get(); }
    uint32_t *clusModuleStart() { return clusModuleStart_d.get(); }

    uint32_t const *moduleStart() const { return moduleStart_d.get(); }
    uint32_t const *clusInModule() const { return clusInModule_d.get(); }
    uint32_t const *moduleId() const { return moduleId_d.get(); }
    uint32_t const *clusModuleStart() const { return clusModuleStart_d.get(); }

    uint32_t const *c_moduleStart() const { return moduleStart_d.get(); }
    uint32_t const *c_clusInModule() const { return clusInModule_d.get(); }
    uint32_t const *c_moduleId() const { return moduleId_d.get(); }
    uint32_t const *c_clusModuleStart() const { return clusModuleStart_d.get(); }

    class DeviceConstView {
    public:
      // TO DO: removed __ldg, check impact on perf with src/cuda.
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t moduleStart(int i) const { return moduleStart_[i]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t clusInModule(int i) const { return clusInModule_[i]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t moduleId(int i) const { return moduleId_[i]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t clusModuleStart(int i) const { return clusModuleStart_[i]; }

      friend SiPixelClustersAlpaka;

      //   private:
      uint32_t const *__restrict__ moduleStart_;
      uint32_t const *__restrict__ clusInModule_;
      uint32_t const *__restrict__ moduleId_;
      uint32_t const *__restrict__ clusModuleStart_;
    };

    const DeviceConstView view() const {
      return DeviceConstView{c_moduleStart(), c_clusInModule(), c_moduleId(), c_clusModuleStart()};
    }

  private:
    cms::alpakatools::device::unique_ptr<uint32_t> moduleStart_d;   // index of the first pixel of each module
    cms::alpakatools::device::unique_ptr<uint32_t> clusInModule_d;  // number of clusters found in each module
    cms::alpakatools::device::unique_ptr<uint32_t> moduleId_d;      // module id of each module

    // originally from rechits
    cms::alpakatools::device::unique_ptr<uint32_t> clusModuleStart_d;  // index of the first cluster of each module

    uint32_t nClusters_h = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
