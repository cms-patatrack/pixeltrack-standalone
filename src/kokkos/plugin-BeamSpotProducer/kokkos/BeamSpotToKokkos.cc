#include "KokkosDataFormats/BeamSpotKokkos.h"

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

namespace KOKKOS_NAMESPACE {
  class BeamSpotToKokkos : public edm::EDProducer {
  public:
    explicit BeamSpotToKokkos(edm::ProductRegistry& reg);
    ~BeamSpotToKokkos() override = default;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    edm::EDPutTokenT<BeamSpotKokkos<KokkosExecSpace>> bsPutToken_;
    // remove bsHost for now
    // typename Kokkos::View<BeamSpotPOD, KokkosExecSpace>::HostMirror bsHost;
  };

  BeamSpotToKokkos::BeamSpotToKokkos(edm::ProductRegistry& reg)
      : bsPutToken_{reg.produces<BeamSpotKokkos<KokkosExecSpace>>()} {}

  void BeamSpotToKokkos::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& bsRaw = iSetup.get<BeamSpotPOD>();
    BeamSpotKokkos<KokkosExecSpace> bs{&bsRaw};

    iEvent.emplace(bsPutToken_, std::move(bs));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(BeamSpotToKokkos);
