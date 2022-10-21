#include "DataFormats/BeamSpotPOD.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

class BeamSpotToPOD : public edm::EDProducer {
public:
  explicit BeamSpotToPOD(edm::ProductRegistry& reg);
  ~BeamSpotToPOD() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  const edm::EDPutTokenT<BeamSpotPOD> bsPutToken_;
};

BeamSpotToPOD::BeamSpotToPOD(edm::ProductRegistry& reg) : bsPutToken_{reg.produces<BeamSpotPOD>()} {}

void BeamSpotToPOD::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iEvent.emplace(bsPutToken_, iSetup.get<BeamSpotPOD>());
}

DEFINE_FWK_MODULE(BeamSpotToPOD);
