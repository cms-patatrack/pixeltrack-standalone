#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernel_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernel_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

namespace pixelgpudetails {

  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel() = default;
    ~PixelRecHitGPUKernel() = default;

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    TrackingRecHit2DCUDA makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                       SiPixelClustersCUDA const& clusters_d,
                                       BeamSpotCUDA const& bs_d,
                                       pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                       cms::cuda::Context const& ctx) const;
  };
}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernel_h
