#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include "KokkosCore/kokkosConfig.h"

#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"
#include "KokkosDataFormats/PixelTrackKokkos.h"

#include "KokkosCore/SimpleVector.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "HelixFitOnGPU.h"

// FIXME  (split header???)
#include "../GPUCACell.h"

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;
}  // namespace edm

namespace KOKKOS_NAMESPACE {
  class CAHitNtupletGeneratorOnGPU {
  public:
    using HitsOnGPU = TrackingRecHit2DSOAView;
    using HitsOnCPU = TrackingRecHit2DKokkos<KokkosExecSpace>;
    using hindex_type = TrackingRecHit2DSOAView::hindex_type;

    using Quality = pixelTrack::Quality;
    using OutputSoA = pixelTrack::TrackSoA;
    using HitContainer = pixelTrack::HitContainer;
    using Tuple = HitContainer;

    using QualityCuts = cAHitNtupletGenerator::QualityCuts;
    using Params = cAHitNtupletGenerator::Params;
    using Counters = cAHitNtupletGenerator::Counters;

  public:
    CAHitNtupletGeneratorOnGPU(edm::ProductRegistry& reg);

    ~CAHitNtupletGeneratorOnGPU();

    Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace> makeTuples(
        TrackingRecHit2DKokkos<KokkosExecSpace> const& hits_d, float bfield, KokkosExecSpace const& execSpace) const;

  private:
#ifdef TODO
    void buildDoublets(HitsOnCPU const& hh, cudaStream_t stream) const;

    void hitNtuplets(HitsOnCPU const& hh, const edm::EventSetup& es, bool useRiemannFit, cudaStream_t cudaStream);

    void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, cudaStream_t cudaStream) const;
#endif

    Params m_params;

    Kokkos::View<Counters, KokkosExecSpace> m_counters;
  };
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
