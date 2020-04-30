#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer3 : public edm::EDProducer {
  public:
    explicit TestProducer3(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

#ifdef TODO
    edm::EDGetTokenT<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>> getToken_;
#endif
  };

  TestProducer3::TestProducer3(edm::ProductRegistry& reg)
#ifdef TODO
      : getToken_(reg.consumes<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>>())
#endif
  {
  }

  void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
#ifdef TODO
    auto const& tmp = event.get(getToken_);
    cms::cuda::ScopedContextProduce ctx(tmp);
#endif
    std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer3);
