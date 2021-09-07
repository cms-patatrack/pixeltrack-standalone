#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDACore/ProduceContext.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "gpuVertexFinder.h"

#undef PIXVERTEX_DEBUG_PRODUCE

class PixelVertexProducerCUDA : public edm::EDProducer {
public:
  explicit PixelVertexProducerCUDA(edm::ProductRegistry& reg);
  ~PixelVertexProducerCUDA() override = default;

private:
  void produceOnGPU(edm::Event& iEvent, const edm::EventSetup& iSetup);
  void produceOnCPU(edm::Event& iEvent, const edm::EventSetup& iSetup);
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  bool onGPU_;

  edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenGPUTrack_;
  edm::EDPutTokenT<ZVertexCUDAProduct> tokenGPUVertex_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenCPUTrack_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenCPUVertex_;

  const gpuVertexFinder::Producer gpuAlgo_;

  // Tracking cuts before sending tracks to vertex algo
  const float ptMin_;
};

PixelVertexProducerCUDA::PixelVertexProducerCUDA(edm::ProductRegistry& reg)
    : onGPU_(true),
      gpuAlgo_(true,   // oneKernel
               true,   // useDensity
               false,  // useDBSCAN
               false,  // useIterative
               2,      // minT
               0.07,   // eps
               0.01,   // errmax
               9       // chi2max
               ),
      ptMin_(0.5)  // 0.5 GeV
{
  if (onGPU_) {
    tokenGPUTrack_ = reg.consumes<cms::cuda::Product<PixelTrackHeterogeneous>>();
    tokenGPUVertex_ = reg.produces<ZVertexCUDAProduct>();
  } else {
    tokenCPUTrack_ = reg.consumes<PixelTrackHeterogeneous>();
    tokenCPUVertex_ = reg.produces<ZVertexHeterogeneous>();
  }
}

void PixelVertexProducerCUDA::produceOnGPU(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::runProduce(iEvent.streamID(), [&](cms::cuda::ProduceContext& ctx) {
    auto const* tracks = ctx.get(iEvent, tokenGPUTrack_).get();

    assert(tracks);

    ctx.emplace(iEvent, tokenGPUVertex_, gpuAlgo_.makeAsync(ctx, tracks, ptMin_));
  });
}

void PixelVertexProducerCUDA::produceOnCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const* tracks = iEvent.get(tokenCPUTrack_).get();
  assert(tracks);

#ifdef PIXVERTEX_DEBUG_PRODUCE
  auto const& tsoa = *tracks;
  auto maxTracks = tsoa.stride();
  std::cout << "size of SoA " << sizeof(tsoa) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits == int(tsoa.hitIndices.size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA for Vertexing at " << tracks << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE

  iEvent.emplace(tokenCPUVertex_, gpuAlgo_.make(tracks, ptMin_));
}

void PixelVertexProducerCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (onGPU_) {
    produceOnGPU(iEvent, iSetup);
  } else {
    produceOnCPU(iEvent, iSetup);
  }
}

DEFINE_FWK_MODULE(PixelVertexProducerCUDA);
