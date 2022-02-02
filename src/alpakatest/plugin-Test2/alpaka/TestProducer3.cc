#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer3 : public edm::EDProducer {
  public:
    explicit TestProducer3(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<cms::alpakatools::device_buffer<Device, float[]>> getToken_;
  };

  TestProducer3::TestProducer3(edm::ProductRegistry& reg)
      : getToken_(reg.consumes<cms::alpakatools::device_buffer<Device, float[]>>()) {}

  void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    const auto& result = event.get(getToken_);

#ifdef SCOPEDCONTEXT
    cms::cuda::ScopedContextProduce ctx(result);
#endif

    std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer3);
