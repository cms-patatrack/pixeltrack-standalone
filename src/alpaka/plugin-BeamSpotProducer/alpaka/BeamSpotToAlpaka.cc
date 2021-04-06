#include "AlpakaDataFormats/BeamSpotAlpaka.h"

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotToAlpaka : public edm::EDProducer {
  public:
    explicit BeamSpotToAlpaka(edm::ProductRegistry& reg);
    ~BeamSpotToAlpaka() override = default;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    edm::EDPutTokenT<BeamSpotAlpaka> bsPutToken_;
  };

  BeamSpotToAlpaka::BeamSpotToAlpaka(edm::ProductRegistry& reg)
      : bsPutToken_{reg.produces<BeamSpotAlpaka>()} {}

  void BeamSpotToAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& bsRaw = iSetup.get<BeamSpotPOD>();

    AlpakaExecSpace space{AlpakaExecSpace()};
    BeamSpotAlpaka bs{&bsRaw, space};

    iEvent.emplace(bsPutToken_, std::move(bs));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(BeamSpotToAlpaka);
