#include <fstream>

#include "CUDADataFormats/BeamSpot.h"
#include "DataFormats/BeamSpotPOD.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

class BeamSpotToCUDA : public edm::EDProducer {
public:
  explicit BeamSpotToCUDA(edm::ProductRegistry& reg);
  ~BeamSpotToCUDA() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDPutTokenT<BeamSpot> bsPutToken_;
};

BeamSpotToCUDA::BeamSpotToCUDA(edm::ProductRegistry& reg) : bsPutToken_(reg.produces<BeamSpot>()) {}

void BeamSpotToCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& bs = iSetup.get<BeamSpotPOD>();

  BeamSpot bsDevice{};
  *(bsDevice.data()) = bs;

  iEvent.emplace(bsPutToken_, std::move(bsDevice));
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);
