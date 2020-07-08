#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#ifdef TODO
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#endif
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"

#include "KokkosCore/kokkosConfig.h"

namespace KOKKOS_NAMESPACE {
  class CAHitNtupletKokkos : public edm::EDProducer {
  public:
    explicit CAHitNtupletKokkos(edm::ProductRegistry& reg);
    ~CAHitNtupletKokkos() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<TrackingRecHit2DKokkos<KokkosExecSpace>> tokenHitGPU_;
#ifdef TODO
    edm::EDPutTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenTrackGPU_;

    CAHitNtupletGeneratorOnGPU gpuAlgo_;
#endif
  };

  CAHitNtupletKokkos::CAHitNtupletKokkos(edm::ProductRegistry& reg)
    : tokenHitGPU_{reg.consumes<TrackingRecHit2DKokkos<KokkosExecSpace>>()}
#ifdef TODO
    ,tokenTrackGPU_{reg.produces<cms::cuda::Product<PixelTrackHeterogeneous>>()},
      gpuAlgo_(reg)
#endif
  {}

  void CAHitNtupletKokkos::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto bf = 0.0114256972711507;  // 1/fieldInGeV

    auto const& hits = iEvent.get(tokenHitGPU_);

#ifdef TODO
    ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits, bf, ctx.stream()));
#endif
  }
}

DEFINE_FWK_KOKKOS_MODULE(CAHitNtupletKokkos);
