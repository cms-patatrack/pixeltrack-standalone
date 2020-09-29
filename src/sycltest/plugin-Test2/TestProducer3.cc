#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "SYCLCore/Product.h"
#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/device_unique_ptr.h"

class TestProducer3 : public edm::EDProducer {
public:
  explicit TestProducer3(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<cms::sycltools::Product<cms::sycltools::device::unique_ptr<float[]>>> getToken_;
};

TestProducer3::TestProducer3(edm::ProductRegistry& reg)
    : getToken_(reg.consumes<cms::sycltools::Product<cms::sycltools::device::unique_ptr<float[]>>>()) {}

void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto const& tmp = event.get(getToken_);
  cms::sycltools::ScopedContextProduce ctx(tmp);
  std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;
}

DEFINE_FWK_MODULE(TestProducer3);
