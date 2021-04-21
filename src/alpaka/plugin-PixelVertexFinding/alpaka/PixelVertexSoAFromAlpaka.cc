#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/ZVertexAlpaka.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelVertexSoAFromAlpaka : public edm::EDProducer {
  public:
    explicit PixelVertexSoAFromAlpaka(edm::ProductRegistry& reg);
    ~PixelVertexSoAFromAlpaka() override = default;

  private:
#ifdef TODO
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
#endif
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    edm::EDGetTokenT<ZVertexAlpaka> tokenAlpaka_;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    edm::EDPutTokenT<ZVertexHost> tokenSOA_;
#endif

#ifdef TODO
    cms::cuda::host::unique_ptr<ZVertexSoA> m_soa;
#endif
  };

  PixelVertexSoAFromAlpaka::PixelVertexSoAFromAlpaka(edm::ProductRegistry& reg)
      : tokenAlpaka_(reg.consumes<ZVertexAlpaka>())
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        ,
        tokenSOA_(reg.produces<ZVertexHost>())
#endif
  {
  }

#ifdef TODO
  void PixelVertexSoAFromAlpaka::acquire(edm::Event const& iEvent,
                                         edm::EventSetup const& iSetup,
                                         edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& inputDataWrapped = iEvent.get(tokenAlpaka_);
    cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    m_soa = inputData.toHostAsync(ctx.stream());
  }
#endif

  void PixelVertexSoAFromAlpaka::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    auto const& inputData = iEvent.get(tokenAlpaka_);
    auto outputData = cms::alpakatools::allocHostBuf<ZVertexSoA>(1u);
    Queue queue(device);
    alpaka::memcpy(queue, outputData, inputData, 1u);
    alpaka::wait(queue);

    // No copies....
    iEvent.emplace(tokenSOA_, std::move(outputData));
#endif
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelVertexSoAFromAlpaka);
