#include <memory>

#include "CUDADataFormats/SiPixelClusters.h"

SiPixelClusters::SiPixelClusters(size_t maxClusters)
    : moduleStart_d{std::make_unique<uint32_t[]>(maxClusters + 1)},
      clusModuleStart_d{std::make_unique<uint32_t[]>(maxClusters + 1)},
      view_d{std::make_unique<DeviceConstView>()},
      clusInModule_d{std::make_unique<uint32_t[]>(maxClusters)},
      moduleId_d{std::make_unique<uint32_t[]>(maxClusters)} {
  view_d->moduleStart_ = moduleStart_d.get();
  view_d->clusInModule_ = clusInModule_d.get();
  view_d->moduleId_ = moduleId_d.get();
  view_d->clusModuleStart_ = clusModuleStart_d.get();
}
