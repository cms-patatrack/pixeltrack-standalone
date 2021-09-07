#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDACore/EDProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/RunningAverage.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

class CAHitNtupletCUDA : public cms::cuda::EDProducer {
public:
  explicit CAHitNtupletCUDA(edm::ProductRegistry& reg);
  ~CAHitNtupletCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup, cms::cuda::ProduceContext& ctx) override;

  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> tokenHitGPU_;
  edm::EDPutTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenTrackGPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletCUDA::CAHitNtupletCUDA(edm::ProductRegistry& reg)
    : tokenHitGPU_{reg.consumes<cms::cuda::Product<TrackingRecHit2DGPU>>()},
      tokenTrackGPU_{reg.produces<cms::cuda::Product<PixelTrackHeterogeneous>>()},
      gpuAlgo_(reg) {}

void CAHitNtupletCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es, cms::cuda::ProduceContext& ctx) {
  auto bf = 0.0114256972711507;  // 1/fieldInGeV

  auto const& hits = ctx.get(iEvent, tokenHitGPU_);

  ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits, bf, ctx));
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);
