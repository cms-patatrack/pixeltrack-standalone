#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClusters_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClusters_h

#include <memory>

class SiPixelClusters {
public:
  SiPixelClusters() = default;
  explicit SiPixelClusters(size_t maxClusters);

  SiPixelClusters(const SiPixelClusters &) = delete;
  SiPixelClusters &operator=(const SiPixelClusters &) = delete;
  SiPixelClusters(SiPixelClusters &&) = default;
  SiPixelClusters &operator=(SiPixelClusters &&) = default;

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
    // DeviceConstView() = default;

    __forceinline__ uint32_t moduleStart(int i) const { return moduleStart_[i]; }
    __forceinline__ uint32_t clusInModule(int i) const { return clusInModule_[i]; }
    __forceinline__ uint32_t moduleId(int i) const { return moduleId_[i]; }
    __forceinline__ uint32_t clusModuleStart(int i) const { return clusModuleStart_[i]; }

    friend SiPixelClusters;

    //   private:
    uint32_t const *moduleStart_;
    uint32_t const *clusInModule_;
    uint32_t const *moduleId_;
    uint32_t const *clusModuleStart_;
  };

  DeviceConstView *view() const { return view_d.get(); }

private:
  std::unique_ptr<uint32_t[]> moduleStart_d;  // index of the first pixel of each module
  // originally from rechits
  std::unique_ptr<uint32_t[]> clusModuleStart_d;  // index of the first cluster of each module

  std::unique_ptr<DeviceConstView> view_d;  // "me" pointer

  std::unique_ptr<uint32_t[]> clusInModule_d;  // number of clusters found in each module
  std::unique_ptr<uint32_t[]> moduleId_d;      // module id of each module

  uint32_t nClusters_h;
};

#endif
