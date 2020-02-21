#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/device_unique_ptr.h"

#include "gpuAlgo1.h"

class TestProducer : public edm::EDProducer {
public:
  explicit TestProducer(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
  edm::EDPutTokenT<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>> putToken_;
};

TestProducer::TestProducer(edm::ProductRegistry& reg)
    : rawGetToken_(reg.consumes<FEDRawDataCollection>()),
      putToken_(reg.produces<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>>()) {}

void TestProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto const value = event.get(rawGetToken_).FEDData(1200).size();
  std::cout << "TestProducer  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
            << eventSetup.get<int>() << " FED 1200 size " << value << std::endl;

  cms::cuda::ScopedContextProduce ctx(event.streamID());

  ctx.emplace(event, putToken_, gpuAlgo1(ctx.stream()));
}

DEFINE_FWK_MODULE(TestProducer);
