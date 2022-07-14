#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

#ifdef CUDAUVM_DISABLE_MANAGED_RECHIT
template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(2001, stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), cudaMemcpyDefault, stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nHits(), stream);
  cudaCheck(
      cudaMemcpyAsync(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), cudaMemcpyDefault, stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int16_t[]>(2 * nHits(), stream);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), cudaMemcpyDefault, stream));
  return ret;
}
#else
template <>
void TrackingRecHit2DCUDA::localCoordToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store32.get(), 4 * nHits(), device, stream));
#endif
}
template <>
void TrackingRecHit2DCUDA::hitsModuleStartToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_hitsModuleStart, 4 * 2001, device, stream));
#endif
}
template <>
void TrackingRecHit2DCUDA::globalCoordToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), device, stream));
#endif
}
template <>
void TrackingRecHit2DCUDA::chargeToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), device, stream));
#endif
}
template <>
void TrackingRecHit2DCUDA::sizeToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), device, stream));
#endif
}

#endif
