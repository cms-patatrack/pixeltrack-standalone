#include <sycl/sycl.hpp>

#include "SYCLCore/Product.h"
#include "SYCLCore/HostProduct.h"

#include "SYCLDataFormats/ZVertexSoA.h"
#include "SYCLDataFormats/HeterogeneousSoA.h"

#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "SYCLCore/ScopedContext.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;

class PixelVertexSoAFromSYCL : public edm::EDProducerExternalWork {
public:
  explicit PixelVertexSoAFromSYCL(edm::ProductRegistry& reg);
  ~PixelVertexSoAFromSYCL() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::sycltools::Product<ZVertexHeterogeneous>> tokenSYCL_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;

  cms::sycltools::host::unique_ptr<ZVertexSoA> m_soa;
};

PixelVertexSoAFromSYCL::PixelVertexSoAFromSYCL(edm::ProductRegistry& reg)
    : tokenSYCL_(reg.consumes<cms::sycltools::Product<ZVertexHeterogeneous>>()),
      tokenSOA_(reg.produces<ZVertexHeterogeneous>()) {}

void PixelVertexSoAFromSYCL::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  auto const& inputDataWrapped = iEvent.get(tokenSYCL_);
  cms::sycltools::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.toHostAsync(ctx.stream());
}

void PixelVertexSoAFromSYCL::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
  iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(m_soa)));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromSYCL);
