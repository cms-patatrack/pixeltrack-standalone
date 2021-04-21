#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/PixelTrackAlpaka.h"

#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#ifdef TODO
  class PixelTrackSoAFromAlpaka : public edm::EDProducerExternalWork {
#else
  class PixelTrackSoAFromAlpaka : public edm::EDProducer {
#endif
  public:
    explicit PixelTrackSoAFromAlpaka(edm::ProductRegistry& reg);
    ~PixelTrackSoAFromAlpaka() override = default;

  private:
#ifdef TODO
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
#endif
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    edm::EDGetTokenT<PixelTrackAlpaka> tokenAlpaka_;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    edm::EDPutTokenT<PixelTrackHost> tokenSOA_;
#endif

#ifdef TODO
    cms::cuda::host::unique_ptr<pixelTrack::TrackSoA> m_soa;
#endif
  };

  PixelTrackSoAFromAlpaka::PixelTrackSoAFromAlpaka(edm::ProductRegistry& reg)
      : tokenAlpaka_(reg.consumes<PixelTrackAlpaka>())
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        ,
        tokenSOA_(reg.produces<PixelTrackHost>())
#endif
  {
  }

#ifdef TODO
  void PixelTrackSoAFromAlpaka::acquire(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    cms::cuda::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenAlpaka_);
    cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    m_soa = inputData.toHostAsync(ctx.stream());
  }
#endif

  void PixelTrackSoAFromAlpaka::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    auto const& inputData = iEvent.get(tokenAlpaka_);
    auto outputData = cms::alpakatools::allocHostBuf<pixelTrack::TrackSoA>(1u);
    Queue queue(device);
    alpaka::memcpy(queue, outputData, inputData, 1u);
    alpaka::wait(queue);

    // DO NOT  make a copy  (actually TWO....)
    iEvent.emplace(tokenSOA_, std::move(outputData));
#endif
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelTrackSoAFromAlpaka);
