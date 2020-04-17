#include <cassert>
#include <future>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "KokkosCore/kokkosConfig.h"

#include "kokkosAlgo2.h"

namespace KOKKOS_NAMESPACE {
  class TestProducer2 : public edm::EDProducer {
  public:
    explicit TestProducer2(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<Kokkos::View<const float*, KokkosExecSpace>> getToken_;
  };

  TestProducer2::TestProducer2(edm::ProductRegistry& reg)
      : getToken_(reg.consumes<Kokkos::View<const float*, KokkosExecSpace>>()) {}

  void TestProducer2::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << std::endl;

    kokkosAlgo2();
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(TestProducer2);
