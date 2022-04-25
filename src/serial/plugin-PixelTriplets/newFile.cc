#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "plugin-SiPixelRecHits/PixelRecHits.h"

// I take a generic file number, just for reference
int test_file = 0;

class myClass : public edm::EDProducer {
public:
  explicit myClass(edm::ProductRegistry& reg);
  ~myClass() override = default;
  
private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  
  // // // Fix algo_ type
  pixelgpudetails::PixelRecHitGPUKernel algo_;
  CAHitNtupletGeneratorOnGPU gpuAlgo_;
  edm::EDPutTokenT<std::vector<float>> test_Token;
  // tokenHitCPU_ should be a PutToken, right?
  edm::EDPutTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
};

myClass::myClass(edm::ProductRegistry& reg)
    : gpuAlgo_(reg),
      test_Token(reg.produces<std::vector<float>>())/*,
      tokenHitCPU_(reg.produces<TrackingRecHit2DCPU>())*/ {}

void myClass::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  std::cout << "I'm here!" << '\n';
  std::vector<float> test = {7,6,5,4,3,2,1};
  iEvent.emplace(test_Token, test);
  iEvent.emplace(tokenHitCPU_, algo_.makeHits(test_file));
}

DEFINE_FWK_MODULE(myClass);
