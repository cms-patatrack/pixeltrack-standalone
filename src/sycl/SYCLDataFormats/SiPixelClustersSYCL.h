#ifndef SYCLDataFormats_SiPixelCluster_interface_SiPixelClustersSYCL_h
#define SYCLDataFormats_SiPixelCluster_interface_SiPixelClustersSYCL_h

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

#include <CL/sycl.hpp>

class SiPixelClustersSYCL {
public:
  SiPixelClustersSYCL() = default;
  explicit SiPixelClustersSYCL(size_t maxClusters, sycl::queue stream);
  ~SiPixelClustersSYCL() = default;

  SiPixelClustersSYCL(const SiPixelClustersSYCL &) = delete;
  SiPixelClustersSYCL &operator=(const SiPixelClustersSYCL &) = delete;
  SiPixelClustersSYCL(SiPixelClustersSYCL &&) = default;
  SiPixelClustersSYCL &operator=(SiPixelClustersSYCL &&) = default;

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

    __attribute__((always_inline)) uint32_t moduleStart(int i) const { return *(moduleStart_ + i); }
    __attribute__((always_inline)) uint32_t clusInModule(int i) const { return *(clusInModule_ + i); }
    __attribute__((always_inline)) uint32_t moduleId(int i) const { return *(moduleId_ + i); }
    __attribute__((always_inline)) uint32_t clusModuleStart(int i) const { return *(clusModuleStart_ + i); }

    friend SiPixelClustersSYCL;

    //   private:
    uint32_t const *moduleStart_;
    uint32_t const *clusInModule_;
    uint32_t const *moduleId_;
    uint32_t const *clusModuleStart_;
  };

  DeviceConstView *view() const { return view_d.get(); }

private:
  cms::sycltools::device::unique_ptr<uint32_t[]> moduleStart_d;   // index of the first pixel of each module
  cms::sycltools::device::unique_ptr<uint32_t[]> clusInModule_d;  // number of clusters found in each module
  cms::sycltools::device::unique_ptr<uint32_t[]> moduleId_d;      // module id of each module

  // originally from rechits
  cms::sycltools::device::unique_ptr<uint32_t[]> clusModuleStart_d;  // index of the first cluster of each module

  cms::sycltools::device::unique_ptr<DeviceConstView> view_d;  // "me" pointer

  uint32_t nClusters_h;
};

#endif
