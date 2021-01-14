#include "KokkosDataFormats/BeamSpotKokkos.h"
#include "KokkosDataFormats/SiPixelClustersKokkos.h"
#include "KokkosDataFormats/SiPixelDigisKokkos.h"
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CondFormats/PixelCPEFast.h"

#include "PixelRecHits.h"  // TODO : spit product from kernel

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"

namespace KOKKOS_NAMESPACE {
  class SiPixelRecHitKokkos : public edm::EDProducer {
  public:
    explicit SiPixelRecHitKokkos(edm::ProductRegistry& reg);
    ~SiPixelRecHitKokkos() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    // The mess with inputs will be cleaned up when migrating to the new framework
    edm::EDGetTokenT<cms::kokkos::Product<BeamSpotKokkos<KokkosExecSpace>>> tBeamSpot;
    edm::EDGetTokenT<cms::kokkos::Product<SiPixelClustersKokkos<KokkosExecSpace>>> token_;
    edm::EDGetTokenT<cms::kokkos::Product<SiPixelDigisKokkos<KokkosExecSpace>>> tokenDigi_;

    edm::EDPutTokenT<cms::kokkos::Product<TrackingRecHit2DKokkos<KokkosExecSpace>>> tokenHit_;

    pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
  };

  SiPixelRecHitKokkos::SiPixelRecHitKokkos(edm::ProductRegistry& reg)
      : tBeamSpot(reg.consumes<cms::kokkos::Product<BeamSpotKokkos<KokkosExecSpace>>>()),
        token_(reg.consumes<cms::kokkos::Product<SiPixelClustersKokkos<KokkosExecSpace>>>()),
        tokenDigi_(reg.consumes<cms::kokkos::Product<SiPixelDigisKokkos<KokkosExecSpace>>>()),
        tokenHit_(reg.produces<cms::kokkos::Product<TrackingRecHit2DKokkos<KokkosExecSpace>>>()) {}

  void SiPixelRecHitKokkos::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto const& fcpe = es.get<PixelCPEFast<KokkosExecSpace>>();

    auto const& pclusters = iEvent.get(token_);
    cms::kokkos::ScopedContextProduce<KokkosExecSpace> ctx{pclusters};

    auto const& bs = ctx.get(iEvent, tBeamSpot);
    auto const& clusters = ctx.get(pclusters);
    auto const& digis = ctx.get(iEvent, tokenDigi_);

    auto nHits = clusters.nClusters();
    if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
      std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
    }

    ctx.emplace(iEvent, tokenHit_, gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.params(), ctx.execSpace()));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(SiPixelRecHitKokkos);
