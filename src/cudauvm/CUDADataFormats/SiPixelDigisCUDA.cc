#include "CUDADataFormats/SiPixelDigisCUDA.h"

#ifndef CUDAUVM_MANAGED_TEMPORARY
#include "CUDACore/device_unique_ptr.h"
#endif
#include "CUDACore/copyAsync.h"
#include "CUDACore/ScopedSetDevice.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream) {
#ifdef CUDAUVM_MANAGED_TEMPORARY
  xx_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
#else
  xx_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
#endif
#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  adc_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::cuda::make_device_unique<int32_t[]>(maxFedWords, stream);
  pdigi_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream);
#else
  adc_d = cms::cuda::make_managed_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::cuda::make_managed_unique<int32_t[]>(maxFedWords, stream);
  pdigi_d = cms::cuda::make_managed_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::cuda::make_managed_unique<uint32_t[]>(maxFedWords, stream);
#endif

#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
#else
  auto view = cms::cuda::make_managed_unique<DeviceConstView>(stream);
#endif
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

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

SiPixelDigisCUDA::~SiPixelDigisCUDA() {
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

#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}

#else  // CUDAUVM_DISABLE_MANAGED_CLUSTERING

void SiPixelDigisCUDA::adcPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(adc_d.get(), nDigis(), device, stream));
#endif
}

void SiPixelDigisCUDA::clusPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(clus_d.get(), nDigis(), device, stream));
#endif
}

void SiPixelDigisCUDA::pdigiPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(pdigi_d.get(), nDigis(), device, stream));
#endif
}

void SiPixelDigisCUDA::rawIdArrPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(rawIdArr_d.get(), nDigis(), device, stream));
#endif
}
#endif  // CUDAUVM_DISABLE_MANAGED_CLUSTERING
