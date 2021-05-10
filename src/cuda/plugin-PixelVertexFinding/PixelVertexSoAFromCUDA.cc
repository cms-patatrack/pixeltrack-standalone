#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDACore/HostProduct.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "CUDACore/ScopedContext.h"

using PixelVertexSoAFromCUDA_AsyncState = cms::cuda::host::unique_ptr<ZVertexSoA>;

class PixelVertexSoAFromCUDA : public edm::EDProducerExternalWork<PixelVertexSoAFromCUDA_AsyncState> {
public:
  explicit PixelVertexSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelVertexSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder,
               AsyncState& state) const override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, AsyncState& state) override;

  edm::EDGetTokenT<cms::cuda::Product<ZVertexHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<ZVertexHeterogeneous>>()),
      tokenSOA_(reg.produces<ZVertexHeterogeneous>()) {}

void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                     AsyncState& state) const {
  auto const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  state = inputData.toHostAsync(ctx.stream());
}

void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup, AsyncState& state) {
  // No copies....
  iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(state)));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
