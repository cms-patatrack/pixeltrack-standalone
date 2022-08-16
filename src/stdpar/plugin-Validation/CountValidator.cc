#include "CUDADataFormats/PixelTrack.h"
#include "CUDADataFormats/SiPixelClusters.h"
#include "CUDADataFormats/SiPixelDigis.h"
#include "CUDADataFormats/ZVertex.h"
#include "DataFormats/DigiClusterCount.h"
#include "DataFormats/TrackCount.h"
#include "DataFormats/VertexCount.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>

namespace {
  std::atomic<int> allEvents = 0;
  std::atomic<int> goodEvents = 0;
  std::atomic<int> sumVertexDifference = 0;

  std::mutex sumTrackDifferenceMutex;
  float sumTrackDifference = 0;
}  // namespace

class CountValidator : public edm::EDProducer {
public:
  explicit CountValidator(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void endJob() override;

  edm::EDGetTokenT<DigiClusterCount> digiClusterCountToken_;
  edm::EDGetTokenT<TrackCount> trackCountToken_;
  edm::EDGetTokenT<VertexCount> vertexCountToken_;

  edm::EDGetTokenT<SiPixelDigis> digiToken_;
  edm::EDGetTokenT<SiPixelClusters> clusterToken_;
  edm::EDGetTokenT<PixelTrack> trackToken_;
  edm::EDGetTokenT<ZVertex> vertexToken_;
};

CountValidator::CountValidator(edm::ProductRegistry& reg)
    : digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
      trackCountToken_(reg.consumes<TrackCount>()),
      vertexCountToken_(reg.consumes<VertexCount>()),
      digiToken_(reg.consumes<SiPixelDigis>()),
      clusterToken_(reg.consumes<SiPixelClusters>()),
      trackToken_(reg.consumes<PixelTrack>()),
      vertexToken_(reg.consumes<ZVertex>()) {}

void CountValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  constexpr float trackTolerance = 0.012f;  // in 200 runs of 1k events all events are withing this tolerance
  constexpr int vertexTolerance = 1;

  std::stringstream ss;
  bool ok = true;

  ss << "Event " << iEvent.eventID() << " ";

  {
    auto const& count = iEvent.get(digiClusterCountToken_);
    auto const& digis = iEvent.get(digiToken_);
    auto const& clusters = iEvent.get(clusterToken_);

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

    int nTracks = 0;
    for (int i = 0; i < tracks->stride(); ++i) {
      if (tracks->nHits(i) > 0) {
        ++nTracks;
      }
    }

    auto rel = std::abs(float(nTracks - int(count.nTracks())) / count.nTracks());
    if (static_cast<unsigned int>(nTracks) != count.nTracks()) {
      std::lock_guard<std::mutex> guard(sumTrackDifferenceMutex);
      sumTrackDifference += rel;
    }
    if (rel >= trackTolerance) {
      ss << "\n N(tracks) is " << nTracks << " expected " << count.nTracks() << ", relative difference " << rel
         << " is outside tolerance " << trackTolerance;
      ok = false;
    }
  }

  {
    auto const& count = iEvent.get(vertexCountToken_);
    auto const& vertices = iEvent.get(vertexToken_);

    auto diff = std::abs(int(vertices->nvFinal) - int(count.nVertices()));
    if (diff != 0) {
      sumVertexDifference += diff;
    }
    if (diff > vertexTolerance) {
      ss << "\n N(vertices) is " << vertices->nvFinal << " expected " << count.nVertices() << ", difference " << diff
         << " is outside tolerance " << vertexTolerance;
      ok = false;
    }
  }

  ++allEvents;
  if (ok) {
    ++goodEvents;
  } else {
    std::cout << ss.str() << std::endl;
  }
}

void CountValidator::endJob() {
  if (allEvents == goodEvents) {
    std::cout << "CountValidator: all " << allEvents << " events passed validation\n";
    if (sumTrackDifference != 0.f) {
      std::cout << " Average relative track difference " << sumTrackDifference / allEvents.load()
                << " (all within tolerance)\n";
    }
    if (sumVertexDifference != 0) {
      std::cout << " Average absolute vertex difference " << float(sumVertexDifference.load()) / allEvents.load()
                << " (all within tolerance)\n";
    }
  } else {
    std::cout << "CountValidator: " << (allEvents - goodEvents) << " events failed validation (see details above)\n";
    throw std::runtime_error("CountValidator failed");
  }
}

DEFINE_FWK_MODULE(CountValidator);
