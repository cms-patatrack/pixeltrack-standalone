#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class TestProducer3 : public edm::EDProducer {
public:
  TestProducer3(edm::ModuleConfig const& config, edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<unsigned int> getToken_;
};

TestProducer3::TestProducer3(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
    : getToken_(reg.consumes<unsigned int>(config.required<std::string>("input"))) {}

void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto const value = event.get(getToken_);
#ifndef PYTEST_SILENT
  std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << " value " << value
            << std::endl;
#endif
}

DEFINE_FWK_MODULE(TestProducer3);
