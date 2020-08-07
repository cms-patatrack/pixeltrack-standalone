#include "DataFormats/ZVertexSoA.h"
#include "KokkosCore/kokkosConfig.h"
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

    edm::EDGetTokenT<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>> tokenTrack_;
    edm::EDPutTokenT<Kokkos::View<ZVertexSoA, KokkosExecSpace>> tokenVertex_;

    const gpuVertexFinder::Producer m_gpuAlgo;

    // Tracking cuts before sending tracks to vertex algo
    const float m_ptMin;
  };

  PixelVertexProducerKokkos::PixelVertexProducerKokkos(edm::ProductRegistry& reg)
      : tokenTrack_(reg.consumes<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>>()),
        tokenVertex_(reg.produces<Kokkos::View<ZVertexSoA, KokkosExecSpace>>()),
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
    auto const& tracks = iEvent.get(tokenTrack_);

    iEvent.emplace(tokenVertex_, m_gpuAlgo.make(tracks, m_ptMin, KokkosExecSpace()));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(PixelVertexProducerKokkos);
