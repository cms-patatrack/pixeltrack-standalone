#include <alpaka/alpaka.hpp>

#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/alpakaMemoryHelper.h"
#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelTrackSoAFromAlpaka : public edm::EDProducerExternalWork {
  public:
    explicit PixelTrackSoAFromAlpaka(edm::ProductRegistry& reg);
    ~PixelTrackSoAFromAlpaka() override = default;

  private:
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    edm::EDGetTokenT<::cms::alpakatools::Product<Queue, PixelTrackAlpaka>> tokenAlpaka_;
    edm::EDPutTokenT<PixelTrackHost> tokenSOA_;

    AlpakaHostBuf<pixelTrack::TrackSoA> soa_;
  };

  PixelTrackSoAFromAlpaka::PixelTrackSoAFromAlpaka(edm::ProductRegistry& reg)
      : tokenAlpaka_(reg.consumes<::cms::alpakatools::Product<Queue, PixelTrackAlpaka>>()),
        tokenSOA_(reg.produces<PixelTrackHost>()),
        soa_{::cms::alpakatools::allocHostBuf<pixelTrack::TrackSoA>(1u)} {}

  void PixelTrackSoAFromAlpaka::acquire(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    soa_ = ::cms::alpakatools::allocHostBuf<pixelTrack::TrackSoA>(1u);
    ::cms::alpakatools::Product<Queue, PixelTrackAlpaka> const& inputDataWrapped = iEvent.get(tokenAlpaka_);
    ::cms::alpakatools::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);
    alpaka::memcpy(ctx.stream(), soa_, inputData, 1u);
  }

  void PixelTrackSoAFromAlpaka::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
    iEvent.emplace(tokenSOA_, std::move(soa_));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelTrackSoAFromAlpaka);
