#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include <cstdint>

#include "KokkosCore/kokkosConfig.h"

#include "KokkosDataFormats/BeamSpotKokkos.h"
#include "KokkosDataFormats/SiPixelClustersKokkos.h"
#include "KokkosDataFormats/SiPixelDigisKokkos.h"
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"

namespace KOKKOS_NAMESPACE {
  namespace pixelgpudetails {
    class PixelRecHitGPUKernel {
    public:
      PixelRecHitGPUKernel() = default;
      ~PixelRecHitGPUKernel() = default;

      PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
      PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
      PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

      TrackingRecHit2DKokkos<KokkosExecSpace> makeHitsAsync(
          SiPixelDigisKokkos<KokkosExecSpace> const& digis_d,
          SiPixelClustersKokkos<KokkosExecSpace> const& clusters_d,
          BeamSpotKokkos<KokkosExecSpace> const& bs_d,
          Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, KokkosExecSpace> const& cpeParams,
          KokkosExecSpace const& execSpace) const;
    };
  }  // namespace pixelgpudetails
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
