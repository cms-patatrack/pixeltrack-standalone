#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include "CUDACore/cudaCompat.h"

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

  PixelTrackHeterogeneous makeTuples(TrackingRecHit2DCPU const& hits_d, float bfield) const;

private:

  Params m_params;

  Counters* m_counters = nullptr;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
