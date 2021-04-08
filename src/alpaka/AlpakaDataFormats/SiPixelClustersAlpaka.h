#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class SiPixelClustersAlpaka {
public:
  SiPixelClustersAlpaka() = default;
explicit SiPixelClustersAlpaka(size_t maxClusters)
  : moduleStart_d{cms::alpakatools::allocDeviceBuf<uint32_t>(device, maxClusters + 1)},
    clusInModule_d{cms::alpakatools::allocDeviceBuf<uint32_t>(device, maxClusters)},
    moduleId_d{cms::alpakatools::allocDeviceBuf<uint32_t>(device, maxClusters)},
    clusModuleStart_d{cms::alpakatools::allocDeviceBuf<uint32_t>(device, maxClusters + 1)}
  {}
  ~SiPixelClustersAlpaka() = default;

  SiPixelClustersAlpaka(const SiPixelClustersAlpaka &) = delete;
  SiPixelClustersAlpaka &operator=(const SiPixelClustersAlpaka &) = delete;
  SiPixelClustersAlpaka(SiPixelClustersAlpaka &&) = default;
  SiPixelClustersAlpaka &operator=(SiPixelClustersAlpaka &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

auto moduleStartAlpakaDeviceBuf() { return moduleStart_d; }

  uint32_t *moduleStart() { return alpaka::getPtrNative(moduleStart_d); }
  uint32_t *clusInModule() { return alpaka::getPtrNative(clusInModule_d); }
  uint32_t *moduleId() { return alpaka::getPtrNative(moduleId_d); }
  uint32_t *clusModuleStart() { return alpaka::getPtrNative(clusModuleStart_d); }

  uint32_t const *moduleStart() const { return alpaka::getPtrNative(moduleStart_d); }
  uint32_t const *clusInModule() const { return alpaka::getPtrNative(clusInModule_d); }
  uint32_t const *moduleId() const { return alpaka::getPtrNative(moduleId_d); }
  uint32_t const *clusModuleStart() const { return alpaka::getPtrNative(clusModuleStart_d); }

  uint32_t const *c_moduleStart() const { return alpaka::getPtrNative(moduleStart_d); }
  uint32_t const *c_clusInModule() const { return alpaka::getPtrNative(clusInModule_d); }
  uint32_t const *c_moduleId() const { return alpaka::getPtrNative(moduleId_d); }
  uint32_t const *c_clusModuleStart() const { return alpaka::getPtrNative(clusModuleStart_d); }

  class DeviceConstView {
public:
    // TO DO: removed __ldg, check impact on perf with src/cuda.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t moduleStart(int i) const { return moduleStart_[i]; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t clusInModule(int i) const { return clusInModule_[i]; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t moduleId(int i) const { return moduleId_[i]; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t clusModuleStart(int i) const { return clusModuleStart_[i]; }

    friend SiPixelClustersAlpaka;

    //   private:
    uint32_t const * __restrict__ moduleStart_;
    uint32_t const * __restrict__ clusInModule_;
    uint32_t const * __restrict__ moduleId_;
    uint32_t const * __restrict__ clusModuleStart_;
  };

const DeviceConstView view() const { return DeviceConstView{c_moduleStart(), c_clusInModule(), c_moduleId(), c_clusModuleStart()}; }

private:
  AlpakaDeviceBuf<uint32_t> moduleStart_d;   // index of the first pixel of each module
  AlpakaDeviceBuf<uint32_t> clusInModule_d;  // number of clusters found in each module
  AlpakaDeviceBuf<uint32_t> moduleId_d;      // module id of each module

  // originally from rechits
  AlpakaDeviceBuf<uint32_t> clusModuleStart_d;  // index of the first cluster of each module

  uint32_t nClusters_h;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
