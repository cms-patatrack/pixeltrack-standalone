#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxModules, cudaStream_t stream)
    : moduleStart_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules + 1, stream)),
      clusInModule_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules, stream)),
      moduleId_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules, stream)),
      clusModuleStart_d(cms::cuda::make_device_unique<uint32_t[]>(maxModules + 1, stream)) {
  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = cms::cuda::make_device_unique<DeviceConstView>(stream);
  cms::cuda::copyAsync(view_d, view, stream);
}
