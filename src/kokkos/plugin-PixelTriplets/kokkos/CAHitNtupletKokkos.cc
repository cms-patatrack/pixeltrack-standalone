#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "KokkosDataFormats/PixelTrackKokkos.h"
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"

namespace KOKKOS_NAMESPACE {
  class CAHitNtupletKokkos : public edm::EDProducer {
  public:
    explicit CAHitNtupletKokkos(edm::ProductRegistry& reg);
    ~CAHitNtupletKokkos() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<cms::kokkos::Product<TrackingRecHit2DKokkos<KokkosDeviceMemSpace>>> tokenHitGPU_;
    edm::EDPutTokenT<cms::kokkos::Product<Kokkos::View<pixelTrack::TrackSoA, KokkosDeviceMemSpace>>> tokenTrackGPU_;

    CAHitNtupletGeneratorOnGPU gpuAlgo_;
  };

  CAHitNtupletKokkos::CAHitNtupletKokkos(edm::ProductRegistry& reg)
      : tokenHitGPU_{reg.consumes<cms::kokkos::Product<TrackingRecHit2DKokkos<KokkosDeviceMemSpace>>>()},
        tokenTrackGPU_{reg.produces<cms::kokkos::Product<Kokkos::View<pixelTrack::TrackSoA, KokkosDeviceMemSpace>>>()},
        gpuAlgo_(reg) {}

  void CAHitNtupletKokkos::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto bf = 0.0114256972711507;  // 1/fieldInGeV

    auto const& phits = iEvent.get(tokenHitGPU_);
    cms::kokkos::ScopedContextProduce<KokkosExecSpace> ctx{phits};
    auto const& hits = ctx.get(phits);

    ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuples(hits, bf, ctx.execSpace()));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(CAHitNtupletKokkos);
