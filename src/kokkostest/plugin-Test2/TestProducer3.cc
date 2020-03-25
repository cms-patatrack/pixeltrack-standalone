#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class TestProducer3 : public edm::EDProducer {
public:
  explicit TestProducer3(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<unsigned int> getToken_;
};

TestProducer3::TestProducer3(edm::ProductRegistry& reg) : getToken_(reg.consumes<unsigned int>()) {}

void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto const value = event.get(getToken_);
  std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << " value " << value
            << std::endl;
}

DEFINE_FWK_MODULE(TestProducer3);
