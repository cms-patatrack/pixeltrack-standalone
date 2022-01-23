#ifndef plugin_PixelTriplets_alpaka_CAHitNtupletGeneratorOnGPU_h
#define plugin_PixelTriplets_alpaka_CAHitNtupletGeneratorOnGPU_h

#include "AlpakaCore/SimpleVector.h"
#include "AlpakaCore/alpakaConfig.h"
//#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "HelixFitOnGPU.h"

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class CAHitNtupletGeneratorOnGPU {
  public:
    using HitsOnGPU = TrackingRecHit2DSoAView;
    using HitsOnCPU = TrackingRecHit2DAlpaka;
    using hindex_type = TrackingRecHit2DSoAView::hindex_type;

    using Quality = pixelTrack::Quality;
    using OutputSoA = pixelTrack::TrackSoA;
    using HitContainer = pixelTrack::HitContainer;
    using Tuple = HitContainer;

    using QualityCuts = cAHitNtupletGenerator::QualityCuts;
    using Params = cAHitNtupletGenerator::Params;
    using Counters = cAHitNtupletGenerator::Counters;

  public:
    CAHitNtupletGeneratorOnGPU(edm::ProductRegistry& reg);
    ~CAHitNtupletGeneratorOnGPU() = default;

    PixelTrackAlpaka makeTuplesAsync(TrackingRecHit2DAlpaka const& hits_d, float bfield, Queue& queue) const;

  private:
#ifdef TODO
    void buildDoublets(HitsOnCPU const& hh, cudaStream_t stream) const;

    void hitNtuplets(HitsOnCPU const& hh, const edm::EventSetup& es, bool useRiemannFit, cudaStream_t cudaStream);

    void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, cudaStream_t cudaStream) const;
#endif

    Params m_params;

    //cms::alpakatools::device_buffer<Device, Counters> m_counters;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_PixelTriplets_alpaka_CAHitNtupletGeneratorOnGPU_h
