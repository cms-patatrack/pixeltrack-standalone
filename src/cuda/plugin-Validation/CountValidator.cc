#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "DataFormats/DigiClusterCount.h"
#include "DataFormats/TrackCount.h"
#include "DataFormats/VertexCount.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include <iostream>
#include <sstream>

class CountValidator : public edm::EDProducer {
public:
  explicit CountValidator(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<DigiClusterCount> digiClusterCountToken_;
  edm::EDGetTokenT<TrackCount> trackCountToken_;
  edm::EDGetTokenT<VertexCount> vertexCountToken_;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiToken_;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterToken_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> trackToken_;
  edm::EDGetTokenT<ZVertexHeterogeneous> vertexToken_;
};

CountValidator::CountValidator(edm::ProductRegistry& reg)
    : digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
      trackCountToken_(reg.consumes<TrackCount>()),
      vertexCountToken_(reg.consumes<VertexCount>()),
      digiToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()),
      trackToken_(reg.consumes<PixelTrackHeterogeneous>()),
      vertexToken_(reg.consumes<ZVertexHeterogeneous>()) {}

void CountValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::stringstream ss;
  bool ok = true;

  ss << "Event " << iEvent.eventID() << " ";

  {
    auto const& pdigis = iEvent.get(digiToken_);
    cms::cuda::ScopedContextProduce ctx{pdigis};
    auto const& count = iEvent.get(digiClusterCountToken_);
    auto const& digis = ctx.get(iEvent, digiToken_);
    auto const& clusters = ctx.get(iEvent, clusterToken_);

    if (digis.nModules() != count.nModules()) {
      ss << "\n N(modules) is " << digis.nModules() << " expected " << count.nModules();
      ok = false;
    }
    if (digis.nDigis() != count.nDigis()) {
      ss << "\n N(digis) is " << digis.nDigis() << " expected " << count.nDigis();
      ok = false;
    }
    if (clusters.nClusters() != count.nClusters()) {
      ss << "\n N(clusters) is " << clusters.nClusters() << " expected " << count.nClusters();
      ok = false;
    }
  }

  {
    auto const& count = iEvent.get(trackCountToken_);
    auto const& tracks = iEvent.get(trackToken_);

    if (tracks->m_nTracks != count.nTracks()) {
      ss << "\n N(tracks) is " << tracks->m_nTracks << " expected " << count.nTracks();
      ok = false;
    }
  }

  {
    auto const& count = iEvent.get(vertexCountToken_);
    auto const& vertices = iEvent.get(vertexToken_);

    if (vertices->nvFinal != count.nVertices()) {
      ss << "\n N(vertices) is " << vertices->nvFinal << " expected " << count.nVertices();
      ok = false;
    }
  }

  if (ok) {
    ss << "OK!";
  }

  std::cout << ss.str() << std::endl;
}

DEFINE_FWK_MODULE(CountValidator);
