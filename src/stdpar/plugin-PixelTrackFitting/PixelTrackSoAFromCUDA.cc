#include "CUDADataFormats/PixelTrack.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

class PixelTrackSoAFromCUDA : public edm::EDProducer {
public:
  explicit PixelTrackSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelTrackSoAFromCUDA() override = default;

private:
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<PixelTrack> tokenCUDA_;
  edm::EDPutTokenT<PixelTrack> tokenSOA_;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<PixelTrack>()), tokenSOA_(reg.produces<PixelTrack>()) {}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
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
  // we have a reference to the unique_ptr owning TrackSoA
  PixelTrack const& track_soa = iEvent.get(tokenCUDA_);
  //Construct a new unique_ptr
  iEvent.emplace(tokenSOA_, std::make_unique<pixelTrack::TrackSoA>(*track_soa));
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
