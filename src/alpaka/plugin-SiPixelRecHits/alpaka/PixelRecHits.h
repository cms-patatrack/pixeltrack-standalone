#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include <cstdint>

#include "AlpakaCore/alpakaCommon.h"

#include "AlpakaDataFormats/BeamSpotAlpaka.h"
#include "AlpakaDataFormats/SiPixelClustersAlpaka.h"
#include "AlpakaDataFormats/SiPixelDigisAlpaka.h"
#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"

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
                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams) const;
    };
  }  // namespace pixelgpudetails

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
