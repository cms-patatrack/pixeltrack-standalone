#ifndef plugin_SiPixelRecHits_alpaka_PixelRecHits_h
#define plugin_SiPixelRecHits_alpaka_PixelRecHits_h

#include "AlpakaCore/config.h"
#include "AlpakaDataFormats/alpaka/BeamSpotAlpaka.h"
#include "AlpakaDataFormats/alpaka/SiPixelClustersAlpaka.h"
#include "AlpakaDataFormats/alpaka/SiPixelDigisAlpaka.h"
#include "AlpakaDataFormats/alpaka/TrackingRecHit2DAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace pixelgpudetails {
    class PixelRecHitGPUKernel {
    public:
      PixelRecHitGPUKernel() = default;
      ~PixelRecHitGPUKernel() = default;

      PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
      PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

      TrackingRecHit2DAlpaka makeHitsAsync(SiPixelDigisAlpaka const& digis_d,
                                           SiPixelClustersAlpaka const& clusters_d,
                                           BeamSpotAlpaka const& bs_d,
                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                           Queue& queue) const;
    };
  }  // namespace pixelgpudetails

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_SiPixelRecHits_alpaka_PixelRecHits_h
