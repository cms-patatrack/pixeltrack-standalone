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

class PixelVertexSoAFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit PixelVertexSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelVertexSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<ZVertexHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;

#ifdef CUDAUVM_DISABLE_MANAGED_VERTEX
  cms::cuda::host::unique_ptr<ZVertexSoA> m_soa;
#else
  const ZVertexSoA* m_soa;
#endif
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<ZVertexHeterogeneous>>()),
      tokenSOA_(reg.produces<ZVertexHeterogeneous>()) {}

void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  auto const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

#ifdef CUDAUVM_DISABLE_MANAGED_VERTEX
  m_soa = inputData.toHostAsync(ctx.stream());
#else
  inputData.prefetchAsync(cudaCpuDeviceId, ctx.stream());
  m_soa = inputData.get();
#endif
}

void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
#ifdef CUDAUVM_DISABLE_MANAGED_VERTEX
  iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(m_soa)));
#else
  iEvent.emplace(tokenSOA_, std::make_unique<ZVertexSoA>(*m_soa));
  m_soa = nullptr;
#endif
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
