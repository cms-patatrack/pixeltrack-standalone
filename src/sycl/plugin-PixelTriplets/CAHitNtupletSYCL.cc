#include <sycl/sycl.hpp>

#include "SYCLCore/Product.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "SYCLCore/ScopedContext.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"
#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"

class CAHitNtupletSYCL : public edm::EDProducer {
public:
  explicit CAHitNtupletSYCL(edm::ProductRegistry& reg);
  ~CAHitNtupletSYCL() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::sycltools::Product<TrackingRecHit2DSYCL>> tokenHitGPU_;
  edm::EDPutTokenT<cms::sycltools::Product<PixelTrackHeterogeneous>> tokenTrackGPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletSYCL::CAHitNtupletSYCL(edm::ProductRegistry& reg)
    : tokenHitGPU_{reg.consumes<cms::sycltools::Product<TrackingRecHit2DSYCL>>()},
      tokenTrackGPU_{reg.produces<cms::sycltools::Product<PixelTrackHeterogeneous>>()},
      gpuAlgo_(reg) {}

void CAHitNtupletSYCL::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  auto bf = 0.0114256972711507;  // 1/fieldInGeV

  auto const& phits = iEvent.get(tokenHitGPU_);
  cms::sycltools::ScopedContextProduce ctx{phits};
  auto const& hits = ctx.get(phits);
  ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits, bf, ctx.stream()));
}

DEFINE_FWK_MODULE(CAHitNtupletSYCL);
