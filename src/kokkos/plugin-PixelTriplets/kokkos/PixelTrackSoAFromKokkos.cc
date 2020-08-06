#include "KokkosCore/kokkosConfig.h"
#include "KokkosDataFormats/PixelTrackKokkos.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

namespace KOKKOS_NAMESPACE {
#ifdef TODO
  class PixelTrackSoAFromKokkos : public edm::EDProducerExternalWork {
#else
  class PixelTrackSoAFromKokkos : public edm::EDProducer {
#endif
  public:
    explicit PixelTrackSoAFromKokkos(edm::ProductRegistry& reg);
    ~PixelTrackSoAFromKokkos() override = default;

  private:
#ifdef TODO
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
#endif
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    using TracksExecSpace = Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>;
    using TracksHostSpace = TracksExecSpace::HostMirror;

    edm::EDGetTokenT<TracksExecSpace> tokenKokkos_;
    edm::EDPutTokenT<TracksHostSpace> tokenSOA_;
#ifdef TODO
    cms::cuda::host::unique_ptr<pixelTrack::TrackSoA> m_soa;
#endif
  };

  PixelTrackSoAFromKokkos::PixelTrackSoAFromKokkos(edm::ProductRegistry& reg)
      : tokenKokkos_(reg.consumes<TracksExecSpace>()), tokenSOA_(reg.produces<TracksHostSpace>()) {}

#ifdef TODO
  void PixelTrackSoAFromKokkos::acquire(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    cms::cuda::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenKokkos_);
    cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    m_soa = inputData.toHostAsync(ctx.stream());
  }
#endif

  void PixelTrackSoAFromKokkos::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
    auto const& inputData = iEvent.get(tokenKokkos_);

    /*
  auto const & tsoa = *m_soa;
  auto maxTracks = tsoa.stride();
  std::cout << "size of SoA" << sizeof(tsoa) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits==int(tsoa.hitIndices.size(it)));
    if (nHits == 0) break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA at " << &tsoa << std::endl;
  */

    TracksHostSpace outputData("tracks");
    Kokkos::deep_copy(KokkosExecSpace(), outputData, inputData);
    KokkosExecSpace().fence();

    // DO NOT  make a copy  (actually TWO....)
    iEvent.emplace(tokenSOA_, std::move(outputData));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(PixelTrackSoAFromKokkos);
