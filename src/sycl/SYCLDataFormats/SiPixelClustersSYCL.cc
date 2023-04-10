#include <CL/sycl.hpp>

#include "SYCLDataFormats/SiPixelClustersSYCL.h"

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

SiPixelClustersSYCL::SiPixelClustersSYCL(size_t maxClusters, sycl::queue stream) {
  moduleStart_d = cms::sycltools::make_device_unique<uint32_t[]>(maxClusters + 1, stream);
  clusInModule_d = cms::sycltools::make_device_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d = cms::sycltools::make_device_unique<uint32_t[]>(maxClusters, stream);
  clusModuleStart_d = cms::sycltools::make_device_unique<uint32_t[]>(maxClusters + 1, stream);

  auto view = cms::sycltools::make_host_unique<DeviceConstView>(stream);
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = cms::sycltools::make_device_unique<DeviceConstView>(stream);
  stream.memcpy(view_d.get(), view.get(), sizeof(DeviceConstView));
}
