#include <cuda_runtime.h>

#include "CUDACore/EDProducer.h"
#include "CUDACore/Product.h"
#include "CUDACore/HostProduct.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/RunningAverage.h"

class PixelVertexSoAFromCUDA : public cms::cuda::SynchronizingEDProducer {
public:
  explicit PixelVertexSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelVertexSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, cms::cuda::AcquireContext& ctx) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, cms::cuda::ProduceContext&) override;

  edm::EDGetTokenT<cms::cuda::Product<ZVertexHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;

  cms::cuda::host::unique_ptr<ZVertexSoA> m_soa;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<ZVertexHeterogeneous>>()),
      tokenSOA_(reg.produces<ZVertexHeterogeneous>()) {}

void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     cms::cuda::AcquireContext& ctx) {
  auto const& inputData = ctx.get(iEvent, tokenCUDA_);

  m_soa = inputData.toHostAsync(ctx.stream());
}

void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup, cms::cuda::ProduceContext&) {
  // No copies....
  iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(m_soa)));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
