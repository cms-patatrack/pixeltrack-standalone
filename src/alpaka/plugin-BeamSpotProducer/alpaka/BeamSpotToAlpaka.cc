#include "AlpakaDataFormats/BeamSpotAlpaka.h"

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"
#include "AlpakaCore/ScopedContext.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotToAlpaka : public edm::EDProducer {
  public:
    explicit BeamSpotToAlpaka(edm::ProductRegistry& reg);
    ~BeamSpotToAlpaka() override = default;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    edm::EDPutTokenT<BeamSpotAlpaka> bsPutToken_;
    // TO DO: Add implementation of cms::alpaka::Product?
    // const edm::EDPutTokenT<::cms::alpaka::Product<BeamSpotAlpaka>> bsPutToken_;
  };

  BeamSpotToAlpaka::BeamSpotToAlpaka(edm::ProductRegistry& reg) : bsPutToken_{reg.produces<BeamSpotAlpaka>()} {}

  void BeamSpotToAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& bsRaw = iSetup.get<BeamSpotPOD>();

    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::ScopedContextProduce ctx{iEvent.streamID()};
    BeamSpotAlpaka bsDevice(&bsRaw, ctx.stream());
    ctx.emplace(iEvent, bsPutToken_, std::move(bsDevice));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(BeamSpotToAlpaka);
