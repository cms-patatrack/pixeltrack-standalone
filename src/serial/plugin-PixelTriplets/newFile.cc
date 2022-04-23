#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

class myClass : public edm::EDProducer {
public:
  explicit myClass(edm::ProductRegistry& reg);
  ~myClass() override = default;
  
private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  //edm::EDGetTokenT<TrackingRecHit2DCPU> tokenHitCPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
  edm::EDPutTokenT<std::vector<float>> test_Token;
};

myClass::myClass(edm::ProductRegistry& reg)
    : gpuAlgo_(reg),
     test_Token(reg.produces<std::vector<float>>()) {}

void myClass::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  std::cout << "I'm here!" << '\n';
  std::vector<float> test = {7,6,5,4,3,2,1};
  iEvent.emplace(test_Token, test);
}

DEFINE_FWK_MODULE(myClass);
