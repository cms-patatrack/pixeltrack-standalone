#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDADataFormats/ZVertex.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "CUDACore/ScopedContext.h"

class PixelVertexSoAFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit PixelVertexSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelVertexSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<ZVertex>> tokenCUDA_;
  edm::EDPutTokenT<ZVertex> tokenSOA_;

  const ZVertexSoA* m_soa;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<ZVertex>>()), tokenSOA_(reg.produces<ZVertex>()) {}

void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  auto const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.get();
}

void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
  iEvent.emplace(tokenSOA_, std::make_unique<ZVertexSoA>(*m_soa));
  m_soa = nullptr;
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
