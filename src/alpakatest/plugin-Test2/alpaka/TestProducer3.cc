#include <any>
#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/EDProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer3 : public EDProducer {
  public:
    explicit TestProducer3(edm::ProductRegistry& reg);

  private:
    void produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) override;

    edm::EDGetTokenT<std::any> getToken_;
  };

  TestProducer3::TestProducer3(edm::ProductRegistry& reg)
      : EDProducer(reg), getToken_(consumes<std::any>()) {}

  void TestProducer3::produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) {
    const auto& result = event.get(getToken_);

    std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer3);
