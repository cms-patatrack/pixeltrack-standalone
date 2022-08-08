#include <fstream>

#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
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
  edm::EDPutTokenT<cms::cuda::Product<BeamSpot>> bsPutToken_;
};

BeamSpotToCUDA::BeamSpotToCUDA(edm::ProductRegistry& reg) : bsPutToken_(reg.produces<cms::cuda::Product<BeamSpot>>()) {}

void BeamSpotToCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& bs = iSetup.get<BeamSpotPOD>();

  cms::cuda::ScopedContextProduce ctx{iEvent.streamID()};

  BeamSpot bsDevice{};
  *(bsDevice.data()) = bs;

  ctx.emplace(iEvent, bsPutToken_, std::move(bsDevice));
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);
