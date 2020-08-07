#include "KokkosCore/kokkosConfig.h"
#include "DataFormats/ZVertexSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

namespace KOKKOS_NAMESPACE {
#ifdef TODO
  class PixelVertexSoAFromKokkos : public edm::EDProducerExternalWork {
#else
  class PixelVertexSoAFromKokkos : public edm::EDProducer {
#endif
  public:
    explicit PixelVertexSoAFromKokkos(edm::ProductRegistry& reg);
    ~PixelVertexSoAFromKokkos() override = default;

  private:
#ifdef TODO
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
#endif
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    using VerticesExecSpace = Kokkos::View<ZVertexSoA, KokkosExecSpace>;
    using VerticesHostSpace = VerticesExecSpace::HostMirror;

    edm::EDGetTokenT<VerticesExecSpace> tokenKokkos_;
    edm::EDPutTokenT<VerticesHostSpace> tokenSOA_;
#ifdef TODO
    cms::cuda::host::unique_ptr<ZVertexSoA> m_soa;
#endif
  };

  PixelVertexSoAFromKokkos::PixelVertexSoAFromKokkos(edm::ProductRegistry& reg)
      : tokenKokkos_(reg.consumes<VerticesExecSpace>()), tokenSOA_(reg.produces<VerticesHostSpace>()) {}

#ifdef TODO
  void PixelVertexSoAFromKokkos::acquire(edm::Event const& iEvent,
                                         edm::EventSetup const& iSetup,
                                         edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& inputDataWrapped = iEvent.get(tokenKokkos_);
    cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    m_soa = inputData.toHostAsync(ctx.stream());
  }
#endif

  void PixelVertexSoAFromKokkos::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
    auto const& inputData = iEvent.get(tokenKokkos_);
    VerticesHostSpace outputData("vertices");
    Kokkos::deep_copy(KokkosExecSpace(), outputData, inputData);
    KokkosExecSpace().fence();
    // No copies....
    iEvent.emplace(tokenSOA_, std::move(outputData));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(PixelVertexSoAFromKokkos);
