#include <sycl/sycl.hpp>

#include "SYCLCore/Product.h"
#include "SYCLCore/HostProduct.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "SYCLCore/ScopedContext.h"

class PixelTrackSoAFromSYCL : public edm::EDProducerExternalWork {
public:
  explicit PixelTrackSoAFromSYCL(edm::ProductRegistry& reg);
  ~PixelTrackSoAFromSYCL() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::sycltools::Product<PixelTrackHeterogeneous>> tokenSYCL_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenSOA_;

  cms::sycltools::host::unique_ptr<pixelTrack::TrackSoA> m_soa;
};

PixelTrackSoAFromSYCL::PixelTrackSoAFromSYCL(edm::ProductRegistry& reg)
    : tokenSYCL_(reg.consumes<cms::sycltools::Product<PixelTrackHeterogeneous>>()),
      tokenSOA_(reg.produces<PixelTrackHeterogeneous>()) {}

void PixelTrackSoAFromSYCL::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::sycltools::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenSYCL_);
  cms::sycltools::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.toHostAsync(ctx.stream());
}

void PixelTrackSoAFromSYCL::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  /*
  auto const & tsoa = *m_soa;
  auto maxTracks = tsoa.stride();
  std::cout << "size of SoA" << sizeof(tsoa) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits==int(tsoa.hitIndices.size(it)));
    if (nHits == 0) break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA at " << &tsoa << std::endl;
  */

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, PixelTrackHeterogeneous(std::move(m_soa)));

  assert(!m_soa);
}

DEFINE_FWK_MODULE(PixelTrackSoAFromSYCL);
