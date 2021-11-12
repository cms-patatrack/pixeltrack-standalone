#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaAlgoConsumer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducerConsume : public edm::EDProducer {
  public:
    explicit TestProducerConsume(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    struct UniqueProduct {};

    edm::EDGetTokenT<TestProduct<Queue>> getToken_;
    edm::EDPutTokenT<UniqueProduct> putToken_;

    AlpakaAlgoConsumer gpuAlgo_;
  };

  TestProducerConsume::TestProducerConsume(edm::ProductRegistry& reg)
    : getToken_(reg.consumes<TestProduct<Queue>>()), putToken_(reg.produces<UniqueProduct>()) {}

  void TestProducerConsume::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto const& input = event.get(getToken_);
    std::cout << "TestProducerConsume  Event " << event.eventID() << " stream " << event.streamID() << std::endl;
    gpuAlgo_.run(input);
    event.emplace(putToken_);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducerConsume);
