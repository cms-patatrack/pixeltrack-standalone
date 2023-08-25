#include <iostream>

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/config.h"
#include "AlpakaDataFormats/alpaka/BeamSpotAlpaka.h"
#include "AlpakaDataFormats/alpaka/SiPixelClustersAlpaka.h"
#include "AlpakaDataFormats/alpaka/SiPixelDigisAlpaka.h"
#include "AlpakaDataFormats/alpaka/TrackingRecHit2DAlpaka.h"
#include "CondFormats/alpaka/PixelCPEFast.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "PixelRecHits.h"  // TODO : split product from kernel

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelRecHitAlpaka : public edm::EDProducer {
  public:
    explicit SiPixelRecHitAlpaka(edm::ProductRegistry& reg);
    ~SiPixelRecHitAlpaka() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    // The mess with inputs will be cleaned up when migrating to the new framework
    edm::EDGetTokenT<cms::alpakatools::Product<Queue, BeamSpotAlpaka>> tBeamSpot;
    edm::EDGetTokenT<cms::alpakatools::Product<Queue, SiPixelClustersAlpaka>> token_;
    edm::EDGetTokenT<cms::alpakatools::Product<Queue, SiPixelDigisAlpaka>> tokenDigi_;

    edm::EDPutTokenT<cms::alpakatools::Product<Queue, TrackingRecHit2DAlpaka>> tokenHit_;

    pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
  };

  SiPixelRecHitAlpaka::SiPixelRecHitAlpaka(edm::ProductRegistry& reg)
      : tBeamSpot(reg.consumes<cms::alpakatools::Product<Queue, BeamSpotAlpaka>>()),
        token_(reg.consumes<cms::alpakatools::Product<Queue, SiPixelClustersAlpaka>>()),
        tokenDigi_(reg.consumes<cms::alpakatools::Product<Queue, SiPixelDigisAlpaka>>()),
        tokenHit_(reg.produces<cms::alpakatools::Product<Queue, TrackingRecHit2DAlpaka>>()) {}

  void SiPixelRecHitAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto const& fcpe = es.get<PixelCPEFast>();

    auto const& pclusters = iEvent.get(token_);
    cms::alpakatools::ScopedContextProduce<Queue> ctx{pclusters};

    auto const& clusters = ctx.get(pclusters);
    auto const& digis = ctx.get(iEvent, tokenDigi_);
    auto const& bs = ctx.get(iEvent, tBeamSpot);

    auto nHits = clusters.nClusters();
    if (nHits >= TrackingRecHit2DSoAView::maxHits()) {
      std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSoAView::maxHits() << std::endl;
    }
    ctx.emplace(iEvent,
                tokenHit_,
                gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.getGPUProductAsync(ctx.stream()), ctx.stream()));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitAlpaka);
