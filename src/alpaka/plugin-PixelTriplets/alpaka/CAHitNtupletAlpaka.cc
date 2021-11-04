#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"
#include "CAHitNtupletGeneratorOnGPU.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAHitNtupletAlpaka : public edm::EDProducer {
  public:
    explicit CAHitNtupletAlpaka(edm::ProductRegistry& reg);
    ~CAHitNtupletAlpaka() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<::cms::alpakatools::Product<Queue, TrackingRecHit2DAlpaka>> tokenHitGPU_;
    edm::EDPutTokenT<::cms::alpakatools::Product<Queue, PixelTrackAlpaka>> tokenTrackGPU_;

    CAHitNtupletGeneratorOnGPU gpuAlgo_;
  };

  CAHitNtupletAlpaka::CAHitNtupletAlpaka(edm::ProductRegistry& reg)
      : tokenHitGPU_{reg.consumes<::cms::alpakatools::Product<Queue, TrackingRecHit2DAlpaka>>()},
        tokenTrackGPU_{reg.produces<::cms::alpakatools::Product<Queue, PixelTrackAlpaka>>()},
        gpuAlgo_(reg) {}

  void CAHitNtupletAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto bf = 0.0114256972711507;  // 1/fieldInGeV

    auto const& phits = iEvent.get(tokenHitGPU_);
    ::cms::alpakatools::ScopedContextProduce<Queue> ctx{phits};
    auto const& hits = ctx.get(phits);

    ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits, bf, ctx.stream()));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpaka);
