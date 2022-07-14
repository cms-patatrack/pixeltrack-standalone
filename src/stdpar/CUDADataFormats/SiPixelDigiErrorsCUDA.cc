#include "CUDADataFormats/SiPixelDigiErrorsCUDA.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/memsetAsync.h"

#include <cassert>

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cudaStream_t stream)
    : formatterErrors_h(std::move(errors)) {
#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  error_d = cms::cuda::make_device_unique<cms::cuda::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cms::cuda::make_device_unique<PixelErrorCompact[]>(maxFedWords, stream);
  cms::cuda::memsetAsync(data_d, 0x00, maxFedWords, stream);

  error_h = cms::cuda::make_host_unique<cms::cuda::SimpleVector<PixelErrorCompact>>(stream);
  cms::cuda::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cms::cuda::copyAsync(error_d, error_h, stream);
#else
  maxFedWords_ = maxFedWords;
  error_d = cms::cuda::make_managed_unique<cms::cuda::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cms::cuda::make_managed_unique<PixelErrorCompact[]>(maxFedWords, stream);
  std::memset(data_d.get(), 0, maxFedWords * sizeof(PixelErrorCompact));

  cms::cuda::make_SimpleVector(error_d.get(), maxFedWords, data_d.get());
  assert(error_d->empty());
  assert(error_d->capacity() == static_cast<int>(maxFedWords));

  auto device = cms::cuda::currentDevice();
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(error_d.get(), sizeof(cms::cuda::SimpleVector<PixelErrorCompact>), device, stream));
  cudaCheck(cudaMemPrefetchAsync(data_d.get(), maxFedWords * sizeof(PixelErrorCompact), device, stream));
#endif
#endif  // CUDAUVM_DISABLE_MANAGED_CLUSTERING
}

#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cudaStream_t stream) {
  cms::cuda::copyAsync(error_h, error_d, stream);
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cudaStream_t stream) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cms::cuda::make_host_unique<PixelErrorCompact[]>(error_h->capacity(), stream);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cms::cuda::copyAsync(data, data_d, error_h->size(), stream);
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(err, std::move(data));
}
#else
void SiPixelDigiErrorsCUDA::prefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(error_d.get(), sizeof(cms::cuda::SimpleVector<PixelErrorCompact>), device, stream));
  cudaCheck(cudaMemPrefetchAsync(data_d.get(), maxFedWords_ * sizeof(PixelErrorCompact), device, stream));
#endif
}
#endif  // CUDAUVM_DISABLE_MANAGED_CLUSTERING
