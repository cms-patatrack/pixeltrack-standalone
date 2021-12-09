#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"
#include "KokkosCore/deep_copy.h"
#include "KokkosCore/shared_ptr.h"
#include "KokkosDataFormats/PixelTrackKokkos.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

namespace KOKKOS_NAMESPACE {
  class PixelTrackSoAFromKokkos : public edm::EDProducerExternalWork {
  public:
    explicit PixelTrackSoAFromKokkos(edm::ProductRegistry& reg);
    ~PixelTrackSoAFromKokkos() override = default;

  private:
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    using TracksDeviceMemSpace = cms::kokkos::shared_ptr<pixelTrack::TrackSoA, KokkosDeviceMemSpace>;
    using TracksHostMemSpace = cms::kokkos::shared_ptr<pixelTrack::TrackSoA, KokkosHostMemSpace>;

    edm::EDGetTokenT<cms::kokkos::Product<TracksDeviceMemSpace>> tokenKokkos_;
    edm::EDPutTokenT<TracksHostMemSpace> tokenSOA_;

    TracksHostMemSpace m_soa;
  };

  PixelTrackSoAFromKokkos::PixelTrackSoAFromKokkos(edm::ProductRegistry& reg)
      : tokenKokkos_(reg.consumes<cms::kokkos::Product<TracksDeviceMemSpace>>()),
        tokenSOA_(reg.produces<TracksHostMemSpace>()) {}

  void PixelTrackSoAFromKokkos::acquire(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& inputDataWrapped = iEvent.get(tokenKokkos_);
    cms::kokkos::ScopedContextAcquire<KokkosExecSpace> ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    m_soa = cms::kokkos::make_shared<pixelTrack::TrackSoA, KokkosHostMemSpace>(ctx.execSpace());
    cms::kokkos::deep_copy(ctx.execSpace(), m_soa, inputData);
  }

  void PixelTrackSoAFromKokkos::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
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

    // DO NOT  make a copy  (actually TWO....)
    iEvent.emplace(tokenSOA_, std::move(m_soa));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(PixelTrackSoAFromKokkos);
