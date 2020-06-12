#include "CUDACore/Product.h"
#include "CUDADataFormats/BeamSpotCUDA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CUDACore/ScopedContext.h"

#include <cuda_runtime.h>

#include <fstream>

class BeamSpotToCUDA : public edm::EDProducer {
public:
  explicit BeamSpotToCUDA(edm::ProductRegistry& reg);
  ~BeamSpotToCUDA() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDPutTokenT<cms::cuda::Product<BeamSpotCUDA>> bsPutToken_;
};

BeamSpotToCUDA::BeamSpotToCUDA(edm::ProductRegistry& reg)
    : bsPutToken_{reg.produces<cms::cuda::Product<BeamSpotCUDA>>()} {}

void BeamSpotToCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& bs = iSetup.get<BeamSpotCUDA::Data>();

  cms::cuda::ScopedContextProduce ctx{iEvent.streamID()};

  ctx.emplace(iEvent, bsPutToken_, bs, ctx.device(), ctx.stream());
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);
