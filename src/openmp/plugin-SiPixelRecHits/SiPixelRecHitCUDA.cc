#include "CUDADataFormats/SiPixelClustersSoA.h"
#include "CUDADataFormats/SiPixelDigisSoA.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "DataFormats/BeamSpotPOD.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CondFormats/PixelCPEFast.h"

#include "PixelRecHits.h"  // TODO : spit product from kernel

class SiPixelRecHitCUDA : public edm::EDProducer {
public:
  explicit SiPixelRecHitCUDA(edm::ProductRegistry& reg);
  ~SiPixelRecHitCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // The mess with inputs will be cleaned up when migrating to the new framework
  edm::EDGetTokenT<BeamSpotPOD> tBeamSpot;
  edm::EDGetTokenT<SiPixelClustersSoA> token_;
  edm::EDGetTokenT<SiPixelDigisSoA> tokenDigi_;

  edm::EDPutTokenT<TrackingRecHit2DCPU> tokenHit_;

  pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
};

SiPixelRecHitCUDA::SiPixelRecHitCUDA(edm::ProductRegistry& reg)
    : tBeamSpot(reg.consumes<BeamSpotPOD>()),
      token_(reg.consumes<SiPixelClustersSoA>()),
      tokenDigi_(reg.consumes<SiPixelDigisSoA>()),
      tokenHit_(reg.produces<TrackingRecHit2DCPU>()) {}

void SiPixelRecHitCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  PixelCPEFast const& fcpe = es.get<PixelCPEFast>();

  auto const& clusters = iEvent.get(token_);
  auto const& digis = iEvent.get(tokenDigi_);
  auto const& bs = iEvent.get(tBeamSpot);

  auto nHits = clusters.nClusters();
  if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
    std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
  }

  iEvent.emplace(tokenHit_, gpuAlgo_.makeHits(digis, clusters, bs, &fcpe.getCPUProduct()));
}

DEFINE_FWK_MODULE(SiPixelRecHitCUDA);
