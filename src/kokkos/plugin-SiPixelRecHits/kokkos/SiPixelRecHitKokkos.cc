#include <cuda_runtime.h>

#include "KokkosDataFormats/BeamSpotKokkos.h"
#include "KokkosDataFormats/SiPixelClustersKokkos.h"
#include "KokkosDataFormats/SiPixelDigisKokkos.h"
#ifdef TODO
#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#endif
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CondFormats/PixelCPEFast.h"

#ifdef TODO
#include "PixelRecHits.h"  // TODO : spit product from kernel
#endif

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

#ifdef TODO
    edm::EDPutTokenT<cms::cuda::Product<TrackingRecHit2DCUDA>> tokenHit_;

    pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
#endif
  };

  SiPixelRecHitKokkos::SiPixelRecHitKokkos(edm::ProductRegistry& reg)
      : tBeamSpot(reg.consumes<BeamSpotKokkos<KokkosExecSpace>>()),
        token_(reg.consumes<SiPixelClustersKokkos<KokkosExecSpace>>()),
        tokenDigi_(reg.consumes<SiPixelDigisKokkos<KokkosExecSpace>>())
#ifdef TODO
        tokenHit_(reg.produces<cms::cuda::Product<TrackingRecHit2DCUDA>>())
#endif
  {}

  void SiPixelRecHitKokkos::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto const& fcpe = es.get<PixelCPEFast<KokkosExecSpace>>();

    auto const& bs = iEvent.get(tBeamSpot);
    auto const& clusters = iEvent.get(token_);
    auto const& digis = iEvent.get(tokenDigi_);

#ifdef TODO
    auto nHits = clusters.nClusters();
    if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
      std::cout << "Clusters/Hits Overflow " << nHits << " >= " << TrackingRecHit2DSOAView::maxHits() << std::endl;
    }

    ctx.emplace(iEvent,
                tokenHit_,
                gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.getGPUProductAsync(ctx.stream()), ctx.stream()));
#endif
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(SiPixelRecHitKokkos);
