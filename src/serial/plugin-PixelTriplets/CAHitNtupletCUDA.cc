#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

class CAHitNtupletCUDA : public edm::EDProducer {
public:
  explicit CAHitNtupletCUDA(edm::ProductRegistry& reg);
  ~CAHitNtupletCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenTrackCPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletCUDA::CAHitNtupletCUDA(edm::ProductRegistry& reg)
    : tokenHitCPU_{reg.consumes<TrackingRecHit2DCPU>()},
      tokenTrackCPU_{reg.produces<PixelTrackHeterogeneous>()},
      gpuAlgo_(reg) {}

void CAHitNtupletCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  auto bf = 0.0114256972711507*2;  // 1/fieldInGeV

  std::cout << "prima del get" << '\n';
  auto const& hits = iEvent.get(tokenHitCPU_);
  std::cout << "x" << hits.view()->xGlobal(0) << '\n';
  std::cout << "y" << hits.view()->yGlobal(0) << '\n';
  std::cout << "z" << hits.view()->zGlobal(0) << '\n';
  
  //auto const& hits_view = hits.view();
  std::cout << "prima di tuples" << '\n';
  PixelTrackHeterogeneous tuples_ = gpuAlgo_.makeTuples(hits, bf);
  std::cout << "dopo tuples" << '\n';
  std::cout << "m_nTracks = " << tuples_->m_nTracks << '\n';
  iEvent.emplace(tokenTrackCPU_, gpuAlgo_.makeTuples(hits, bf));
  std::cout << "--------------------------------------------------------------------------------" << '\n';
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);
