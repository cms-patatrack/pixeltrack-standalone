#include <memory>

#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "gpuVertexFinder.h"

class PixelVertexProducerCUDA : public edm::EDProducer {
public:
  explicit PixelVertexProducerCUDA(edm::ProductRegistry& reg);
  ~PixelVertexProducerCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<PixelTrack> tokenTrack_;
  edm::EDPutTokenT<ZVertex> tokenVertex_;

  const gpuVertexFinder::Producer m_gpuAlgo;

  // Tracking cuts before sending tracks to vertex algo
  const float m_ptMin;
};

PixelVertexProducerCUDA::PixelVertexProducerCUDA(edm::ProductRegistry& reg)
    : tokenTrack_{reg.consumes<PixelTrack>()},
      tokenVertex_{reg.produces<ZVertex>()},
      m_gpuAlgo(true,   // oneKernel
                true,   // useDensity
                false,  // useDBSCAN
                false,  // useIterative
                2,      // minT
                0.07,   // eps
                0.01,   // errmax
                9       // chi2max
                ),
      m_ptMin(0.5)  // 0.5 GeV
      {}

void PixelVertexProducerCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const* tracks = iEvent.get(tokenTrack_).get();

  assert(tracks);
  //move unique_ptr into local variable
  auto vertices{m_gpuAlgo.makeAsync(tracks, m_ptMin)};
  cudaDeviceSynchronize(); //wait for the device to finish kernels
  //We now move the unique_ptr into the event
  iEvent.emplace(tokenVertex_, std::move(vertices));
}

DEFINE_FWK_MODULE(PixelVertexProducerCUDA);
