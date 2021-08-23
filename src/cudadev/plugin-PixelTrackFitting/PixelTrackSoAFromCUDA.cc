#include <cuda_runtime.h>

#include "CUDACore/Context.h"
#include "CUDACore/EDProducer.h"
#include "CUDACore/Product.h"
#include "CUDACore/HostProduct.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

// Switch on to enable checks and printout for found tracks
#undef PIXEL_DEBUG_PRODUCE

class PixelTrackSoAFromCUDA : public cms::cuda::SynchronizingEDProducer {
public:
  explicit PixelTrackSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelTrackSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, cms::cuda::AcquireContext& ctx) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, cms::cuda::ProduceContext&) override;

  edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenSOA_;

  cms::cuda::host::unique_ptr<pixelTrack::TrackSoA> soa_;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<PixelTrackHeterogeneous>>()),
      tokenSOA_(reg.produces<PixelTrackHeterogeneous>()) {}

void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    cms::cuda::AcquireContext& ctx) {
  auto const& inputData = ctx.get(iEvent, tokenCUDA_);

  soa_ = inputData.toHostAsync(ctx.stream());
}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup, cms::cuda::ProduceContext&) {
#ifdef PIXEL_DEBUG_PRODUCE
  auto const& tsoa = *soa_;
  auto maxTracks = tsoa.stride();
  std::cout << "size of SoA" << sizeof(tsoa) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits == int(tsoa.hitIndices.size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA at " << &tsoa << std::endl;
#endif

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, PixelTrackHeterogeneous(std::move(soa_)));

  assert(!soa_);
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
