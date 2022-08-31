#include <cuda_runtime.h>

#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrack.h"
#include "CUDADataFormats/TrackingRecHit2D.h"

class CAHitNtupletCUDA : public edm::EDProducer {
public:
  explicit CAHitNtupletCUDA(edm::ProductRegistry& reg);
  ~CAHitNtupletCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<TrackingRecHit2D> tokenHitGPU_;
  edm::EDPutTokenT<PixelTrack> tokenTrackGPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletCUDA::CAHitNtupletCUDA(edm::ProductRegistry& reg)
    : tokenHitGPU_{reg.consumes<TrackingRecHit2D>()},
      tokenTrackGPU_{reg.produces<PixelTrack>()},
      gpuAlgo_(reg) {}

void CAHitNtupletCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  auto bf = 0.0114256972711507;  // 1/fieldInGeV

  auto const& hits = iEvent.get(tokenHitGPU_);

  iEvent.emplace(tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits, bf));
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);
