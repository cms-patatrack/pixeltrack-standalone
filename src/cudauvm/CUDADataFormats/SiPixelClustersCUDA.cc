#include "CUDADataFormats/SiPixelClustersCUDA.h"

#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/ScopedSetDevice.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxClusters, cudaStream_t stream) {
#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  moduleStart_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters + 1, stream);
  clusModuleStart_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters + 1, stream);
#else
  moduleStart_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters + 1, stream);
  clusModuleStart_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters + 1, stream);
#endif
#ifdef CUDAUVM_MANAGED_TEMPORARY
  clusInModule_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d = cms::cuda::make_managed_unique<uint32_t[]>(maxClusters, stream);
#else
  clusInModule_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
#endif

#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
#else
  auto view = cms::cuda::make_managed_unique<DeviceConstView>(stream);
#endif
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  view_d = cms::cuda::make_device_unique<DeviceConstView>(stream);
  cms::cuda::copyAsync(view_d, view, stream);
#else
  view_d = std::move(view);
  device_ = cms::cuda::currentDevice();
#ifndef CUDAUVM_DISABLE_ADVISE
  cudaCheck(cudaMemAdvise(view_d.get(), sizeof(DeviceConstView), cudaMemAdviseSetReadMostly, device_));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(view_d.get(), sizeof(DeviceConstView), device_, stream));
#endif
#endif  // CUDAUVM_DISABLE_MANAGED_CLUSTERING
}

SiPixelClustersCUDA::~SiPixelClustersCUDA() {
#ifndef CUDAUVM_DISABLE_MANAGED_CLUSTERING
#ifndef CUDAUVM_DISABLE_ADVISE
  if (view_d) {
    // need to make sure a CUDA context is initialized for a thread
    cms::cuda::ScopedSetDevice(0);
    cudaCheck(cudaMemAdvise(view_d.get(), sizeof(DeviceConstView), cudaMemAdviseUnsetReadMostly, device_));
  }
#endif
#endif
}
