#include "DataFormats/ZVertexSoA.h"
#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"
#include "KokkosDataFormats/PixelTrackKokkos.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  class PixelVertexProducerKokkos : public edm::EDProducer {
  public:
    explicit PixelVertexProducerKokkos(edm::ProductRegistry& reg);
    ~PixelVertexProducerKokkos() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<cms::kokkos::Product<Kokkos::View<pixelTrack::TrackSoA, KokkosDeviceMemSpace>>> tokenTrack_;
    edm::EDPutTokenT<cms::kokkos::Product<Kokkos::View<ZVertexSoA, KokkosDeviceMemSpace>>> tokenVertex_;

    const gpuVertexFinder::Producer m_gpuAlgo;

    // Tracking cuts before sending tracks to vertex algo
    const float m_ptMin;
  };

  PixelVertexProducerKokkos::PixelVertexProducerKokkos(edm::ProductRegistry& reg)
      : tokenTrack_(reg.consumes<cms::kokkos::Product<Kokkos::View<pixelTrack::TrackSoA, KokkosDeviceMemSpace>>>()),
        tokenVertex_(reg.produces<cms::kokkos::Product<Kokkos::View<ZVertexSoA, KokkosDeviceMemSpace>>>()),
        m_gpuAlgo(true,   // oneKernel
                  true,   // useDensity
                  false,  // useDBSCAN
                  false,  // useIterative
                  2,      // minT
                  0.07,   // eps
                  0.01,   // errmax
                  9       // chi2max
                  ),
        m_ptMin(0.5)  // 0.5 GeV
  {}

  void PixelVertexProducerKokkos::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& ptracks = iEvent.get(tokenTrack_);
    cms::kokkos::ScopedContextProduce<KokkosExecSpace> ctx{ptracks};
    auto const& tracks = ctx.get(ptracks);

    ctx.emplace(iEvent, tokenVertex_, m_gpuAlgo.make(tracks, m_ptMin, ctx.execSpace()));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(PixelVertexProducerKokkos);
