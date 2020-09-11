#include <cassert>
#include <future>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"

#include "kokkosAlgo2.h"

namespace {
  std::atomic<int> nevents;
}

namespace KOKKOS_NAMESPACE {
  class TestProducer2 : public edm::EDProducerExternalWork {
  public:
    explicit TestProducer2(edm::ProductRegistry& reg);

  private:
    void acquire(edm::Event const& event,
                 edm::EventSetup const& eventSetup,
                 edm::WaitingTaskWithArenaHolder holder) override;
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
    void endJob() override;

    edm::EDGetTokenT<cms::kokkos::Product<Kokkos::View<const float*, KokkosExecSpace>>> getToken_;
  };

  TestProducer2::TestProducer2(edm::ProductRegistry& reg)
      : getToken_(reg.consumes<cms::kokkos::Product<Kokkos::View<const float*, KokkosExecSpace>>>()) {
    nevents = 0;
  }

  void TestProducer2::acquire(edm::Event const& event,
                              edm::EventSetup const& eventSetup,
                              edm::WaitingTaskWithArenaHolder holder) {
    auto const& tmp = event.get(getToken_);

    cms::kokkos::ScopedContextAcquire<KokkosExecSpace> ctx(tmp, std::move(holder));

    auto const& array = ctx.get(tmp);
    kokkosAlgo2(ctx.execSpace());

    std::cout << "TestProducer2::acquire Event " << event.eventID() << " stream " << event.streamID() << " array "
              << array.data() << std::endl;
  }

  void TestProducer2::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << std::endl;
    ++nevents;
  }

  void TestProducer2::endJob() {
    std::cout << "TestProducer2::endJob processed " << nevents.load() << " events" << std::endl;
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(TestProducer2);
