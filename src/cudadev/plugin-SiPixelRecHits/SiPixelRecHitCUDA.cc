#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpotCUDA.h"
#include "CUDACore/EDProducer.h"
#include "CUDACore/Product.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "CondFormats/PixelCPEFast.h"

#include "PixelRecHitGPUKernel.h"

class SiPixelRecHitCUDA : public cms::cuda::EDProducer {
public:
  explicit SiPixelRecHitCUDA(edm::ProductRegistry& reg);
  ~SiPixelRecHitCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup, cms::cuda::ProduceContext& ctx) override;

  // The mess with inputs will be cleaned up when migrating to the new framework
  const edm::EDGetTokenT<cms::cuda::Product<BeamSpotCUDA>> tBeamSpot;
  const edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> token_;
  const edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> tokenDigi_;
  const edm::EDPutTokenT<cms::cuda::Product<TrackingRecHit2DCUDA>> tokenHit_;
  const pixelgpudetails::PixelRecHitGPUKernel gpuAlgo_;
};

SiPixelRecHitCUDA::SiPixelRecHitCUDA(edm::ProductRegistry& reg)
    : tBeamSpot(reg.consumes<cms::cuda::Product<BeamSpotCUDA>>()),
      token_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()),
      tokenDigi_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      tokenHit_(reg.produces<cms::cuda::Product<TrackingRecHit2DCUDA>>()) {}

void SiPixelRecHitCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es, cms::cuda::ProduceContext& ctx) {
  PixelCPEFast const& fcpe = es.get<PixelCPEFast>();

  auto const& clusters = ctx.get(iEvent, token_);
  auto const& digis = ctx.get(iEvent, tokenDigi_);
  auto const& bs = ctx.get(iEvent, tBeamSpot);

  ctx.emplace(iEvent,
              tokenHit_,
              gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.getGPUProductAsync(ctx.stream()), ctx.stream()));
}

DEFINE_FWK_MODULE(SiPixelRecHitCUDA);
