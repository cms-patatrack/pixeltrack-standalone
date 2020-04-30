#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"

#include "alpakaAlgo2.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer2 : public edm::EDProducerExternalWork {
  public:
    explicit TestProducer2(edm::ProductRegistry& reg);

  private:
    void acquire(edm::Event const& event,
                 edm::EventSetup const& eventSetup,
                 edm::WaitingTaskWithArenaHolder holder) override;
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

#ifdef TODO
    edm::EDGetTokenT<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>> getToken_;
#endif
  };

  TestProducer2::TestProducer2(edm::ProductRegistry& reg)
#ifdef TODO
      : getToken_(reg.consumes<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>>())
#endif
  {
  }

  void TestProducer2::acquire(edm::Event const& event,
                              edm::EventSetup const& eventSetup,
                              edm::WaitingTaskWithArenaHolder holder) {
#ifdef TODO
    auto const& tmp = event.get(getToken_);

    cms::cuda::ScopedContextAcquire ctx(tmp, std::move(holder));

    auto const& array = ctx.get(tmp);
#endif
    alpakaAlgo2();

    std::cout << "TestProducer2::acquire Event " << event.eventID() << " stream " << event.streamID()
#ifdef TODO
              << " array " << array.get()
#endif
              << std::endl;
  }

  void TestProducer2::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer2);
