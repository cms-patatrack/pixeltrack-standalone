#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "KokkosDataFormats/PixelTrackKokkos.h"
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
    edm::EDPutTokenT<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>> tokenTrackGPU_;

    CAHitNtupletGeneratorOnGPU gpuAlgo_;
  };

  CAHitNtupletKokkos::CAHitNtupletKokkos(edm::ProductRegistry& reg)
      : tokenHitGPU_{reg.consumes<TrackingRecHit2DKokkos<KokkosExecSpace>>()},
        tokenTrackGPU_{reg.produces<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>>()},
        gpuAlgo_(reg) {}

  void CAHitNtupletKokkos::produce(edm::Event& iEvent, const edm::EventSetup& es) {
    auto bf = 0.0114256972711507;  // 1/fieldInGeV

    auto const& hits = iEvent.get(tokenHitGPU_);

    iEvent.emplace(tokenTrackGPU_, gpuAlgo_.makeTuples(hits, bf, KokkosExecSpace()));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(CAHitNtupletKokkos);
