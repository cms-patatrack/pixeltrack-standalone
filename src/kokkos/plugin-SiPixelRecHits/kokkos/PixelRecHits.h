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

      TrackingRecHit2DKokkos<KokkosDeviceMemSpace> makeHitsAsync(
          SiPixelDigisKokkos<KokkosDeviceMemSpace> const& digis_d,
          SiPixelClustersKokkos<KokkosDeviceMemSpace> const& clusters_d,
          BeamSpotKokkos<KokkosDeviceMemSpace> const& bs_d,
          Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, KokkosDeviceMemSpace> const& cpeParams,
          KokkosExecSpace const& execSpace) const;
    };
  }  // namespace pixelgpudetails
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
