#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxModules, cms::cuda::Context const& ctx)
    : moduleStart_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules + 1, ctx)),
      clusInModule_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules, ctx)),
      moduleId_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules, ctx)),
      clusModuleStart_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules + 1, ctx)) {
  auto view = cms::cuda::make_host_unique<DeviceConstView>(ctx);
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = cms::cuda::make_device_unique<DeviceConstView>(ctx);
  cms::cuda::copyAsync(view_d, view, ctx.stream());
}
