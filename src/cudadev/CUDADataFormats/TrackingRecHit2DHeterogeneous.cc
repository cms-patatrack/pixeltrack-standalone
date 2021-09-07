#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), ctx);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), ctx.stream());
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(
    cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(gpuClustering::maxNumModules + 1, ctx);
  cudaCheck(cudaMemcpyAsync(ret.get(),
                            m_hitsModuleStart,
                            sizeof(uint32_t) * (gpuClustering::maxNumModules + 1),
                            cudaMemcpyDefault,
                            ctx.stream()));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), ctx);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), cudaMemcpyDefault, ctx.stream()));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nHits(), ctx);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), cudaMemcpyDefault, ctx.stream()));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<int16_t[]>(2 * nHits(), ctx);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), cudaMemcpyDefault, ctx.stream()));
  return ret;
}
