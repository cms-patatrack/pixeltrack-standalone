#include "CUDADataFormats/SiPixelClustersSoA.h"

SiPixelClustersSoA::SiPixelClustersSoA(size_t maxClusters) {
  moduleStart_d = std::make_unique<uint32_t[]>(maxClusters + 1);
  clusInModule_d = std::make_unique<uint32_t[]>(maxClusters);
  moduleId_d = std::make_unique<uint32_t[]>(maxClusters);
  clusModuleStart_d = std::make_unique<uint32_t[]>(maxClusters + 1);

  auto view = std::make_unique<DeviceConstView>();
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = std::move(view);
}
