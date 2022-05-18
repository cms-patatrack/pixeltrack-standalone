#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "PixelRecHitsCustom.h"

// I take a generic file number, just for reference
int test_file = 6000;

class myClass : public edm::EDProducer {
public:
  explicit myClass(edm::ProductRegistry& reg);
  ~myClass() override = default;
  
private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  
  pixelgpudetails::PixelRecHitGPUKernelCustom algo_;
  CAHitNtupletGeneratorOnGPU gpuAlgo_;
  edm::EDPutTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
};

myClass::myClass(edm::ProductRegistry& reg)
    : algo_(),
      gpuAlgo_(reg),
      tokenHitCPU_(reg.produces<TrackingRecHit2DCPU>()) {}

void myClass::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  iEvent.emplace(tokenHitCPU_, algo_.makeHits2(test_file));
  std::cout << "tutto ok nel producer" << '\n';
}

DEFINE_FWK_MODULE(myClass);
