#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include <cstdint>

#include "DataFormats/BeamSpotPOD.h"
#include "CUDADataFormats/SiPixelClustersSoA.h"
#include "CUDADataFormats/SiPixelDigisSoA.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

namespace pixelgpudetails {

  class PixelRecHitGPUKernelCustom {
  public:
    PixelRecHitGPUKernelCustom() = default;
    ~PixelRecHitGPUKernelCustom() = default;

    PixelRecHitGPUKernelCustom(const PixelRecHitGPUKernelCustom&) = delete;
    PixelRecHitGPUKernelCustom(PixelRecHitGPUKernelCustom&&) = delete;
    PixelRecHitGPUKernelCustom& operator=(const PixelRecHitGPUKernelCustom&) = delete;
    PixelRecHitGPUKernelCustom& operator=(PixelRecHitGPUKernelCustom&&) = delete;

    TrackingRecHit2DCPU makeHits(SiPixelDigisSoA const& digis_d, SiPixelClustersSoA const& clusters_d,
              BeamSpotPOD const& bs_d, pixelCPEforGPU::ParamsOnGPU const* cpeParams) const;
    TrackingRecHit2DCPU makeHits2(int filename_number) const;
  };
}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
