#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/EDProducer.h"

#include "alpakaAlgo2.h"

namespace {
  std::atomic<int> nevents;
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer2 : public EDProducerExternalWork {
  public:
    explicit TestProducer2(edm::ProductRegistry& reg);

  private:
    void acquire(Event const& event,
                 edm::EventSetup const& eventSetup,
                 Context& ctx) override;
    void produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) override;
    void endJob() override;

    edm::EDGetTokenT<std::any> getToken_;
  };

  TestProducer2::TestProducer2(edm::ProductRegistry& reg)
    : EDProducerExternalWork(reg), getToken_(consumes<std::any>()) {
    nevents = 0;
  }

  void TestProducer2::acquire(Event const& event,
                              edm::EventSetup const& eventSetup,
                              Context& ctx) {
    auto const& array = event.get(getToken_);
    alpakaAlgo2(ctx.queue());

    std::cout << "TestProducer2::acquire Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  }

  void TestProducer2::produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) {
    std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << std::endl;
    ++nevents;
  }

  void TestProducer2::endJob() {
    std::cout << "TestProducer2::endJob processed " << nevents.load() << " events" << std::endl;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer2);
