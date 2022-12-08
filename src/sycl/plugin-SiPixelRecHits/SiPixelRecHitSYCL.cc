#include <CL/sycl.hpp>

#include "SYCLDataFormats/BeamSpotSYCL.h"
#include "SYCLCore/Product.h"
#include "SYCLDataFormats/SiPixelClustersSYCL.h"
#include "SYCLDataFormats/SiPixelDigisSYCL.h"
#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "SYCLCore/ScopedContext.h"
#include "CondFormats/PixelCPEFast.h"

#include "PixelRecHits.h"  // TODO : spit product from kernel

class SiPixelRecHitSYCL : public edm::EDProducer {
public:
  explicit SiPixelRecHitSYCL(edm::ProductRegistry& reg);
  ~SiPixelRecHitSYCL() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // The mess with inputs will be cleaned up when migrating to the new framework
  edm::EDGetTokenT<cms::sycltools::Product<BeamSpotSYCL>> tBeamSpot;
  edm::EDGetTokenT<cms::sycltools::Product<SiPixelClustersSYCL>> token_;
  edm::EDGetTokenT<cms::sycltools::Product<SiPixelDigisSYCL>> tokenDigi_;

  edm::EDPutTokenT<cms::sycltools::Product<TrackingRecHit2DSYCL>> tokenHit_;

  pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
};

SiPixelRecHitSYCL::SiPixelRecHitSYCL(edm::ProductRegistry& reg)
    : tBeamSpot(reg.consumes<cms::sycltools::Product<BeamSpotSYCL>>()),
      token_(reg.consumes<cms::sycltools::Product<SiPixelClustersSYCL>>()),
      tokenDigi_(reg.consumes<cms::sycltools::Product<SiPixelDigisSYCL>>()),
      tokenHit_(reg.produces<cms::sycltools::Product<TrackingRecHit2DSYCL>>()) {}

void SiPixelRecHitSYCL::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  PixelCPEFast const& fcpe = es.get<PixelCPEFast>();

  auto const& pclusters = iEvent.get(token_);
  cms::sycltools::ScopedContextProduce ctx{pclusters};

  auto const& clusters = ctx.get(pclusters);
  auto const& digis = ctx.get(iEvent, tokenDigi_);
  auto const& bs = ctx.get(iEvent, tBeamSpot);

  auto nHits = clusters.nClusters();
  if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
    std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
  }

  ctx.emplace(iEvent,
              tokenHit_,
              gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.getGPUProductAsync(ctx.stream()), ctx.stream()));
}

DEFINE_FWK_MODULE(SiPixelRecHitSYCL);
