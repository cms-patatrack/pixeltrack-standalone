#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include <cstdint>

#include <CL/sycl.hpp>

#include "SYCLDataFormats/BeamSpotSYCL.h"
#include "SYCLDataFormats/SiPixelClustersSYCL.h"
#include "SYCLDataFormats/SiPixelDigisSYCL.h"
#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"

namespace pixelgpudetails {

  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel() = default;
    ~PixelRecHitGPUKernel() = default;

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    TrackingRecHit2DSYCL makeHitsAsync(SiPixelDigisSYCL const& digis_d,
                                       SiPixelClustersSYCL const& clusters_d,
                                       BeamSpotSYCL const& bs_d,
                                       pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                       sycl::queue stream) const;
  };
}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
