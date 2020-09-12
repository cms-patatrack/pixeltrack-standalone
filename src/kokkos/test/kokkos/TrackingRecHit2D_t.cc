#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#include "CondFormats/pixelCPEforGPU.h"
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"

namespace testTrackingRecHit2DKokkos {

  template<typename MemorySpace>
  void fill(Kokkos::View<TrackingRecHit2DSOAView, MemorySpace> hits) {

    assert(hits.data());
    auto &hits_ = *hits.data();

    Kokkos::parallel_for("fill", Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1024), KOKKOS_LAMBDA(const size_t i) {
              assert(hits_.nHits() == 200);
              if (i > 200) return;
            });

    return;
  }

  template<typename MemorySpace>
  void verify(Kokkos::View<TrackingRecHit2DSOAView, MemorySpace> hits) {

    assert(hits.data());

    auto const &hits_ = *hits.data();

    Kokkos::parallel_for("fill", Kokkos::RangePolicy<KokkosExecSpace>(0, 1024), KOKKOS_LAMBDA(const size_t i) {
              assert(hits_.nHits() == 200);
              if (i > 200) return;
            });

    return;
  }

  template<typename MemorySpace>
  void runKernels(Kokkos::View<TrackingRecHit2DSOAView, MemorySpace> hits) {
    assert(hits.data());

    fill(hits);
    verify(hits);    
  }
}  // namespace testTrackingRecHit2DKokkos

namespace testTrackingRecHit2DKokkos {
  template<typename MemorySpace>  
  void runKernels(Kokkos::View<TrackingRecHit2DSOAView, MemorySpace> hits);

}

int main() {	

  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});

  {
    auto nHits = 200;

    Kokkos::View<pixelCPEforGPU::ParamsOnGPU, KokkosExecSpace> _cpeParams("cpeparams");
    Kokkos::View<uint32_t*, KokkosExecSpace> _hitsModuleStart("hitsmodulestart");

    TrackingRecHit2DKokkos<KokkosExecSpace> tkhit(nHits, _cpeParams, _hitsModuleStart, KokkosExecSpace());

    testTrackingRecHit2DKokkos::runKernels<KokkosExecSpace>(tkhit.mView());

  }

  return 0;
}
