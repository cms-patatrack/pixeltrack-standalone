#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/device_unique_ptr.h"

class TestProducer3 : public edm::EDProducer {
public:
  explicit TestProducer3(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>> getToken_;
};

TestProducer3::TestProducer3(edm::ProductRegistry& reg)
    : getToken_(reg.consumes<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>>()) {}

void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto const& tmp = event.get(getToken_);
  cms::cuda::ScopedContextProduce ctx(tmp);
  std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;
}

DEFINE_FWK_MODULE(TestProducer3);
