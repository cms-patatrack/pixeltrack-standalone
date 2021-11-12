#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaAlgoProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducerProduce : public edm::EDProducer {
  public:
    explicit TestProducerProduce(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
    edm::EDPutTokenT<TestProduct<Queue>> putToken_;

    AlpakaAlgoProducer gpuAlgo_;
  };

  TestProducerProduce::TestProducerProduce(edm::ProductRegistry& reg)
    : rawGetToken_(reg.consumes<FEDRawDataCollection>()), putToken_(reg.produces<TestProduct<Queue>>()) {}

  void TestProducerProduce::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto const value = event.get(rawGetToken_).FEDData(1200).size();
    std::cout << "TestProducerProduce  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
              << eventSetup.get<int>() << " FED 1200 size " << value << std::endl;

    event.emplace(putToken_, gpuAlgo_.run());
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducerProduce);
