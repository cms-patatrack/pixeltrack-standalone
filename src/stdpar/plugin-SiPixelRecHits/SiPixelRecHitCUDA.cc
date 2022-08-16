#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot.h"
#include "CUDACore/Product.h"
#include "CUDADataFormats/SiPixelClusters.h"
#include "CUDADataFormats/SiPixelDigis.h"
#include "CUDADataFormats/TrackingRecHit2D.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CUDACore/ScopedContext.h"
#include "CondFormats/PixelCPEFast.h"

#include "PixelRecHits.h"  // TODO : spit product from kernel

class SiPixelRecHitCUDA : public edm::EDProducer {
public:
  explicit SiPixelRecHitCUDA(edm::ProductRegistry& reg);
  ~SiPixelRecHitCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // The mess with inputs will be cleaned up when migrating to the new framework
  edm::EDGetTokenT<cms::cuda::Product<BeamSpot>> tBeamSpot;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelClusters>> token_;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigis>> tokenDigi_;

  edm::EDPutTokenT<cms::cuda::Product<TrackingRecHit2D>> tokenHit_;

  pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
};

SiPixelRecHitCUDA::SiPixelRecHitCUDA(edm::ProductRegistry& reg)
    : tBeamSpot(reg.consumes<cms::cuda::Product<BeamSpot>>()),
      token_(reg.consumes<cms::cuda::Product<SiPixelClusters>>()),
      tokenDigi_(reg.consumes<cms::cuda::Product<SiPixelDigis>>()),
      tokenHit_(reg.produces<cms::cuda::Product<TrackingRecHit2D>>()) {}

void SiPixelRecHitCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  PixelCPEFast const& fcpe = es.get<PixelCPEFast>();

  auto const& pclusters = iEvent.get(token_);
  cms::cuda::ScopedContextProduce ctx{pclusters};

  auto const& clusters = ctx.get(pclusters);
  auto const& digis = ctx.get(iEvent, tokenDigi_);
  auto const& bs = ctx.get(iEvent, tBeamSpot);

  auto nHits = clusters.nClusters();
  if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
    std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
  }

  ctx.emplace(iEvent, tokenHit_, gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.get(), ctx.stream()));
}

DEFINE_FWK_MODULE(SiPixelRecHitCUDA);
