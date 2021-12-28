
#include "AlpakaCore/alpakaCommon.h"

#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/ZVertexAlpaka.h"

#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "AlpakaCore/ScopedContext.h"

#include "gpuVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelVertexProducerAlpaka : public edm::EDProducer {
  public:
    explicit PixelVertexProducerAlpaka(edm::ProductRegistry& reg);
    ~PixelVertexProducerAlpaka() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<cms::alpakatools::Product<Queue, PixelTrackAlpaka>> tokenTrack_;
    edm::EDPutTokenT<cms::alpakatools::Product<Queue, ZVertexAlpaka>> tokenVertex_;

    const gpuVertexFinder::Producer m_gpuAlgo;

    // Tracking cuts before sending tracks to vertex algo
    const float m_ptMin;
  };

  PixelVertexProducerAlpaka::PixelVertexProducerAlpaka(edm::ProductRegistry& reg)
      : tokenTrack_(reg.consumes<cms::alpakatools::Product<Queue, PixelTrackAlpaka>>()),
        tokenVertex_(reg.produces<cms::alpakatools::Product<Queue, ZVertexAlpaka>>()),
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

  void PixelVertexProducerAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    cms::alpakatools::Product<Queue, PixelTrackAlpaka> const& tracksBufWrapped = iEvent.get(tokenTrack_);
    cms::alpakatools::ScopedContextProduce<Queue> ctx{tracksBufWrapped};
    auto const& tracksBuf = ctx.get(tracksBufWrapped);
    auto const tracks = alpaka::getPtrNative(tracksBuf);
    ctx.emplace(iEvent, tokenVertex_, m_gpuAlgo.makeAsync(tracks, m_ptMin, ctx.stream()));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelVertexProducerAlpaka);
