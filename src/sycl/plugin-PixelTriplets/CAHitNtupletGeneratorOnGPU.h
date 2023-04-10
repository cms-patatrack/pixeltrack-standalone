#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include <CL/sycl.hpp>

#include "SYCLCore/SimpleVector.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"
#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"

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
  using HitsOnCPU = TrackingRecHit2DSYCL;
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

  ~CAHitNtupletGeneratorOnGPU() = default;

  PixelTrackHeterogeneous makeTuplesAsync(TrackingRecHit2DSYCL const& hits_d, float bfield, sycl::queue stream);

private:
  void buildDoublets(HitsOnCPU const& hh, sycl::queue stream) const;

  void hitNtuplets(HitsOnCPU const& hh, const edm::EventSetup& es, bool useRiemannFit, sycl::queue stream);

  void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, sycl::queue stream) const;

  Params m_params;

  cms::sycltools::device::unique_ptr<Counters> m_counters;
  // In CUDA it is initialized as a raw pointer and then assigned in the constructor
  // In SYCL we need the queue, so we have to:
  // 1) declare it as a unique_ptr to be able to do make_device_unique and to use the DeviceDeleter in cms::sycltools::device::impl
  // 2) initialize it in the scope of makeTupleAsync (for this reason makeTupleAsync cannot be marked const as it was in CUDA)
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h