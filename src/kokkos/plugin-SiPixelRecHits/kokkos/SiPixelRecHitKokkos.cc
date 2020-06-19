#include <cuda_runtime.h>

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

namespace KOKKOS_NAMESPACE {
  class SiPixelRecHitKokkos : public edm::EDProducer {
  public:
    explicit SiPixelRecHitKokkos(edm::ProductRegistry& reg);
    ~SiPixelRecHitKokkos() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    // The mess with inputs will be cleaned up when migrating to the new framework
    edm::EDGetTokenT<BeamSpotKokkos<KokkosExecSpace>> tBeamSpot;
    edm::EDGetTokenT<SiPixelClustersKokkos<KokkosExecSpace>> token_;
    edm::EDGetTokenT<SiPixelDigisKokkos<KokkosExecSpace>> tokenDigi_;

    edm::EDPutTokenT<TrackingRecHit2DKokkos<KokkosExecSpace>> tokenHit_;

    pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
  };

  SiPixelRecHitKokkos::SiPixelRecHitKokkos(edm::ProductRegistry& reg)
      : tBeamSpot(reg.consumes<BeamSpotKokkos<KokkosExecSpace>>()),
        token_(reg.consumes<SiPixelClustersKokkos<KokkosExecSpace>>()),
        tokenDigi_(reg.consumes<SiPixelDigisKokkos<KokkosExecSpace>>()),
        tokenHit_(reg.produces<TrackingRecHit2DKokkos<KokkosExecSpace>>()) {}

  void SiPixelRecHitKokkos::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto const& fcpe = es.get<PixelCPEFast<KokkosExecSpace>>();

    auto const& bs = iEvent.get(tBeamSpot);
    auto const& clusters = iEvent.get(token_);
    auto const& digis = iEvent.get(tokenDigi_);

    auto nHits = clusters.nClusters();
    if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
      std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
    }

    iEvent.emplace(tokenHit_, gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.params(), KokkosExecSpace()));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(SiPixelRecHitKokkos);
