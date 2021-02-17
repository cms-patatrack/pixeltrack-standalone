#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include <hip/hip_runtime.h>

#include "CUDACore/SimpleVector.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DCUDA.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "HelixFitOnGPU.h"

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;
}  // namespace edm

class CAHitNtupletGeneratorOnGPU {
public:
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DCUDA;
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

  PixelTrackHeterogeneous makeTuplesAsync(TrackingRecHit2DGPU const& hits_d, float bfield, hipStream_t stream) const;

  PixelTrackHeterogeneous makeTuples(TrackingRecHit2DCPU const& hits_d, float bfield) const;

private:
  void buildDoublets(HitsOnCPU const& hh, hipStream_t stream) const;

  void hitNtuplets(HitsOnCPU const& hh, const edm::EventSetup& es, bool useRiemannFit, hipStream_t cudaStream);

  void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, hipStream_t cudaStream) const;

  Params m_params;

  Counters* m_counters = nullptr;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
