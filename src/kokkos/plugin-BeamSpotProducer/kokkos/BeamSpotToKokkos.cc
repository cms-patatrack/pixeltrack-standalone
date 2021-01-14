#include "KokkosDataFormats/BeamSpotKokkos.h"

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"

namespace KOKKOS_NAMESPACE {
  class BeamSpotToKokkos : public edm::EDProducer {
  public:
    explicit BeamSpotToKokkos(edm::ProductRegistry& reg);
    ~BeamSpotToKokkos() override = default;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    edm::EDPutTokenT<cms::kokkos::Product<BeamSpotKokkos<KokkosExecSpace>>> bsPutToken_;
    // remove bsHost for now
    // typename Kokkos::View<BeamSpotPOD, KokkosExecSpace>::HostMirror bsHost;
  };

  BeamSpotToKokkos::BeamSpotToKokkos(edm::ProductRegistry& reg)
      : bsPutToken_{reg.produces<cms::kokkos::Product<BeamSpotKokkos<KokkosExecSpace>>>()} {}

  void BeamSpotToKokkos::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& bsRaw = iSetup.get<BeamSpotPOD>();
    cms::kokkos::ScopedContextProduce<KokkosExecSpace> ctx;
    BeamSpotKokkos<KokkosExecSpace> bs{&bsRaw, ctx.execSpace()};

    ctx.emplace(iEvent, bsPutToken_, std::move(bs));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(BeamSpotToKokkos);
