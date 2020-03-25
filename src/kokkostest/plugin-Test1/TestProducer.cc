#include <iostream>
#include <thread>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

class TestProducer : public edm::EDProducer {
public:
  explicit TestProducer(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
  edm::EDPutTokenT<unsigned int> putToken_;
};

TestProducer::TestProducer(edm::ProductRegistry& reg)
    : rawGetToken_(reg.consumes<FEDRawDataCollection>()), putToken_(reg.produces<unsigned int>()) {}

void TestProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto const value = event.get(rawGetToken_).FEDData(1200).size();
  std::cout << "TestProducer  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
            << eventSetup.get<int>() << " FED 1200 size " << value << std::endl;
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(10ms);
  event.emplace(putToken_, static_cast<unsigned int>(event.eventID() + 10 * event.streamID() + 100));
}

DEFINE_FWK_MODULE(TestProducer);
