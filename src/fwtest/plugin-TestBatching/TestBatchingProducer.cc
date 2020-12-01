#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventRange.h"
#include "Framework/PluginFactory.h"

class TestBatchingProducer : public edm::EDBatchingProducer {
public:
  explicit TestBatchingProducer(edm::ProductRegistry& reg);

private:
  void produce(edm::EventRange events, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<unsigned int> getToken_;
};

TestBatchingProducer::TestBatchingProducer(edm::ProductRegistry& reg) : getToken_(reg.consumes<unsigned int>()) {}

void TestBatchingProducer::produce(edm::EventRange events, edm::EventSetup const& eventSetup) {
  for (edm::Event& event : events) {
    auto const value = event.get(getToken_);
#ifndef FWTEST_SILENT
    std::cout << "TestBatchingProducer Event " << event.eventID() << " stream " << event.streamID() << " value "
              << value << std::endl;
#endif
  }
}

DEFINE_FWK_MODULE(TestBatchingProducer);
