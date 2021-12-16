#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/EDProducer.h"

#include "alpakaAlgo1.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer : public EDProducer {
  public:
    explicit TestProducer(edm::ProductRegistry& reg);

  private:
    void produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) override;

    edm::EDGetTokenT<edm::Host<FEDRawDataCollection>> rawGetToken_;
    edm::EDPutTokenT<std::any> putToken_;
  };

  TestProducer::TestProducer(edm::ProductRegistry& reg)
      : EDProducer(reg),
        rawGetToken_(consumes<edm::Host<FEDRawDataCollection>>()),
        putToken_(produces<std::any>()) {}

  void TestProducer::produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) {
    auto const value = event.get(rawGetToken_).FEDData(1200).size();
    std::cout << "TestProducer  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
              << eventSetup.get<int>() << " FED 1200 size " << value << std::endl;

    event.emplace(putToken_, alpakaAlgo1(ctx.queue()));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer);
