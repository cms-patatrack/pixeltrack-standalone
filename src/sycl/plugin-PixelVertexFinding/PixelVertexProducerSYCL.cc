#include <CL/sycl.hpp>

#include "SYCLCore/Product.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "SYCLCore/ScopedContext.h"

#include "gpuVertexFinder.h"

class PixelVertexProducerSYCL : public edm::EDProducer {
public:
  explicit PixelVertexProducerSYCL(edm::ProductRegistry& reg);
  ~PixelVertexProducerSYCL() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::sycltools::Product<PixelTrackHeterogeneous>> tokenGPUTrack_;
  edm::EDPutTokenT<cms::sycltools::Product<ZVertexHeterogeneous>> tokenGPUVertex_;
  //edm::EDGetTokenT<PixelTrackHeterogeneous> tokenCPUTrack_;
  //edm::EDPutTokenT<ZVertexHeterogeneous> tokenCPUVertex_;

  const gpuVertexFinder::Producer m_gpuAlgo;

  // Tracking cuts before sending tracks to vertex algo
  const float m_ptMin;
  std::optional<bool> isCpu_;
};

PixelVertexProducerSYCL::PixelVertexProducerSYCL(edm::ProductRegistry& reg)
    : m_gpuAlgo(true,   // oneKernel
                true,   // useDensity
                false,  // useDBSCAN
                false,  // useIterative
                2,      // minT
                0.07,   // eps
                0.01,   // errmax
                9       // chi2max
                ),
      m_ptMin(0.5)  // 0.5 GeV
{
  tokenGPUTrack_ = reg.consumes<cms::sycltools::Product<PixelTrackHeterogeneous>>();
  tokenGPUVertex_ = reg.produces<cms::sycltools::Product<ZVertexHeterogeneous>>();
}

void PixelVertexProducerSYCL::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& ptracks = iEvent.get(tokenGPUTrack_);

  cms::sycltools::ScopedContextProduce ctx{ptracks};
  auto const* tracks = ctx.get(ptracks).get();

  if (!isCpu_)
    isCpu_ = ctx.stream().get_device().is_cpu();

  assert(tracks);

  ctx.emplace(iEvent, tokenGPUVertex_, m_gpuAlgo.makeAsync(ctx.stream(), tracks, m_ptMin, *isCpu_));
}

DEFINE_FWK_MODULE(PixelVertexProducerSYCL);
