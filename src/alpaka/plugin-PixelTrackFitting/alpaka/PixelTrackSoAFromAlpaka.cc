#include <utility>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/alpaka/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/PixelTrackHost.h"
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

    edm::EDGetTokenT<cms::alpakatools::Product<Queue, PixelTrackAlpaka>> tokenDevice_;
    edm::EDPutTokenT<PixelTrackHost> tokenHost_;

    cms::alpakatools::host_buffer<pixelTrack::TrackSoA> soa_;
  };

  PixelTrackSoAFromAlpaka::PixelTrackSoAFromAlpaka(edm::ProductRegistry& reg)
      : tokenDevice_(reg.consumes<cms::alpakatools::Product<Queue, PixelTrackAlpaka>>()),
        tokenHost_(reg.produces<PixelTrackHost>()),
        soa_{cms::alpakatools::make_host_buffer<pixelTrack::TrackSoA, Platform>()} {}

  void PixelTrackSoAFromAlpaka::acquire(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    cms::alpakatools::Product<Queue, PixelTrackAlpaka> const& inputDataWrapped = iEvent.get(tokenDevice_);
    cms::alpakatools::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);
    soa_ = cms::alpakatools::make_host_buffer<pixelTrack::TrackSoA>(ctx.stream());
    alpaka::memcpy(ctx.stream(), soa_, inputData);
  }

  void PixelTrackSoAFromAlpaka::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
    iEvent.emplace(tokenHost_, std::move(soa_));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelTrackSoAFromAlpaka);
