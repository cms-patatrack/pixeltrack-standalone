#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "KokkosCore/kokkosConfig.h"

namespace KOKKOS_NAMESPACE {
  class TestProducer3 : public edm::EDProducer {
  public:
    explicit TestProducer3(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<Kokkos::View<const float*, KokkosExecSpace>> getToken_;
  };

  TestProducer3::TestProducer3(edm::ProductRegistry& reg)
      : getToken_(reg.consumes<Kokkos::View<const float*, KokkosExecSpace>>()) {}

  void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(TestProducer3);
