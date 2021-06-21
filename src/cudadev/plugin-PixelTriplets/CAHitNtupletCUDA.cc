#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "CUDACore/ScopedContext.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

class CAHitNtupletCUDA : public edm::EDProducer {
public:
  explicit CAHitNtupletCUDA(edm::ProductRegistry& reg);
  ~CAHitNtupletCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> tokenHitGPU_;
  edm::EDPutTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenTrackGPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletCUDA::CAHitNtupletCUDA(edm::ProductRegistry& reg)
    : tokenHitGPU_{reg.consumes<cms::cuda::Product<TrackingRecHit2DGPU>>()},
      tokenTrackGPU_{reg.produces<cms::cuda::Product<PixelTrackHeterogeneous>>()},
      gpuAlgo_(reg) {}

void CAHitNtupletCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  auto bf = 0.0114256972711507;  // 1/fieldInGeV

  auto const& phits = iEvent.get(tokenHitGPU_);
  cms::cuda::ScopedContextProduce ctx{phits};
  auto const& hits = ctx.get(phits);

  ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits, bf, ctx.stream()));
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);
