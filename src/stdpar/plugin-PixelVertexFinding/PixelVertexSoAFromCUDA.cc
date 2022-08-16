#include "CUDADataFormats/ZVertex.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

class PixelVertexSoAFromCUDA : public edm::EDProducer {
public:
  explicit PixelVertexSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelVertexSoAFromCUDA() override = default;

private:
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<ZVertex> tokenCUDA_;
  edm::EDPutTokenT<ZVertex> tokenSOA_;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<ZVertex>()), tokenSOA_(reg.produces<ZVertex>()) {}

void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
  auto const& soa = iEvent.get(tokenCUDA_);
  iEvent.emplace(tokenSOA_, std::make_unique<ZVertexSoA>(*soa));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
