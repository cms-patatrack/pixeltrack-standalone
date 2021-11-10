#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaAlgoIsolatedMember.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducerIsolatedMember : public edm::EDProducer {
  public:
    explicit TestProducerIsolatedMember(edm::ProductRegistry& reg);

    struct UniqueOutput {};

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
    edm::EDPutTokenT<UniqueOutput> putToken_;

    AlpakaAlgoIsolatedMember gpuAlgo_;
  };

  TestProducerIsolatedMember::TestProducerIsolatedMember(edm::ProductRegistry& reg)
    : rawGetToken_(reg.consumes<FEDRawDataCollection>()), putToken_(reg.produces<UniqueOutput>()) {}

  void TestProducerIsolatedMember::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto const value = event.get(rawGetToken_).FEDData(1200).size();
    std::cout << "TestProducerIsolatedMember  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
              << eventSetup.get<int>() << " FED 1200 size " << value << std::endl;

    gpuAlgo_.run();
    event.emplace(putToken_);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducerIsolatedMember);
