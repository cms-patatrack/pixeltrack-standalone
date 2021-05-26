#include "AlpakaDataFormats/BeamSpotAlpaka.h"
#include "AlpakaDataFormats/SiPixelClustersAlpaka.h"
#include "AlpakaDataFormats/SiPixelDigisAlpaka.h"
#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CondFormats/PixelCPEFast.h"

#include "PixelRecHits.h"  // TODO : spit product from kernel

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelRecHitAlpaka : public edm::EDProducer {
  public:
    explicit SiPixelRecHitAlpaka(edm::ProductRegistry& reg);
    ~SiPixelRecHitAlpaka() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    // The mess with inputs will be cleaned up when migrating to the new framework
    edm::EDGetTokenT<BeamSpotAlpaka> tBeamSpot;
    edm::EDGetTokenT<SiPixelClustersAlpaka> token_;
    edm::EDGetTokenT<SiPixelDigisAlpaka> tokenDigi_;

    edm::EDPutTokenT<TrackingRecHit2DAlpaka> tokenHit_;

    pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
  };

  SiPixelRecHitAlpaka::SiPixelRecHitAlpaka(edm::ProductRegistry& reg)
      : tBeamSpot(reg.consumes<BeamSpotAlpaka>()),
        token_(reg.consumes<SiPixelClustersAlpaka>()),
        tokenDigi_(reg.consumes<SiPixelDigisAlpaka>()),
        tokenHit_(reg.produces<TrackingRecHit2DAlpaka>()) {}

  void SiPixelRecHitAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto const& fcpe = es.get<PixelCPEFast>();

    auto const& bs = iEvent.get(tBeamSpot);
    auto const& clusters = iEvent.get(token_);
    auto const& digis = iEvent.get(tokenDigi_);

    auto nHits = clusters.nClusters();
    if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
      std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
    }

    // TO DO: Async: Would need to add a queue as a parameter, not async for now!
    iEvent.emplace(tokenHit_, gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.params()));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitAlpaka);
