#include <cuda_runtime.h>
#include <utility>

#include "CUDADataFormats/BeamSpot.h"
#include "CUDADataFormats/SiPixelClusters.h"
#include "CUDADataFormats/SiPixelDigis.h"
#include "CUDADataFormats/TrackingRecHit2D.h"
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
  edm::EDGetTokenT<BeamSpot> tBeamSpot;
  edm::EDGetTokenT<SiPixelClusters> token_;
  edm::EDGetTokenT<SiPixelDigis> tokenDigi_;

  edm::EDPutTokenT<TrackingRecHit2D> tokenHit_;

  pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
};

SiPixelRecHitCUDA::SiPixelRecHitCUDA(edm::ProductRegistry& reg)
    : tBeamSpot(reg.consumes<BeamSpot>()),
      token_(reg.consumes<SiPixelClusters>()),
      tokenDigi_(reg.consumes<SiPixelDigis>()),
      tokenHit_(reg.produces<TrackingRecHit2D>()) {}

void SiPixelRecHitCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  PixelCPEFast const& fcpe = es.get<PixelCPEFast>();

  auto const& clusters = iEvent.get(token_);
  auto const& digis = iEvent.get(tokenDigi_);
  auto const& bs = iEvent.get(tBeamSpot);

  auto nHits = clusters.nClusters();
  if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
    std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
  }
  auto recHits2D{gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.get())};
  cudaDeviceSynchronize();
  iEvent.emplace(tokenHit_, std::move(recHits2D));
}

DEFINE_FWK_MODULE(SiPixelRecHitCUDA);
