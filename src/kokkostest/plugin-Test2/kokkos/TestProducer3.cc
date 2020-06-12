#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"

namespace KOKKOS_NAMESPACE {
  class TestProducer3 : public edm::EDProducer {
  public:
    explicit TestProducer3(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<cms::kokkos::Product<Kokkos::View<const float*, KokkosExecSpace>>> getToken_;
  };

  TestProducer3::TestProducer3(edm::ProductRegistry& reg)
      : getToken_(reg.consumes<cms::kokkos::Product<Kokkos::View<const float*, KokkosExecSpace>>>()) {}

  void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    cms::kokkos::ScopedContextProduce<KokkosExecSpace> ctx{event.get(getToken_)};
    std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(TestProducer3);
