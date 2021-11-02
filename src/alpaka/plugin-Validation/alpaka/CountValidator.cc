#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/SiPixelClustersAlpaka.h"
#include "AlpakaDataFormats/SiPixelDigisAlpaka.h"
#include "AlpakaDataFormats/ZVertexAlpaka.h"
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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CountValidator : public edm::EDProducer {
  public:
    explicit CountValidator(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endJob() override;

    edm::EDGetTokenT<DigiClusterCount> digiClusterCountToken_;
    edm::EDGetTokenT<TrackCount> trackCountToken_;
    edm::EDGetTokenT<VertexCount> vertexCountToken_;

    edm::EDGetTokenT<SiPixelDigisAlpaka> digiToken_;
    edm::EDGetTokenT<SiPixelClustersAlpaka> clusterToken_;
    edm::EDGetTokenT<PixelTrackHost> trackToken_;
    edm::EDGetTokenT<ZVertexHost> vertexToken_;

    static std::atomic<int> allEvents;
    static std::atomic<int> goodEvents;
    static std::atomic<int> sumVertexDifference;

    static std::mutex sumTrackDifferenceMutex;
    static float sumTrackDifference;
  };

  std::atomic<int> CountValidator::allEvents{0};
  std::atomic<int> CountValidator::goodEvents{0};
  std::atomic<int> CountValidator::sumVertexDifference{0};
  std::mutex CountValidator::sumTrackDifferenceMutex;
  float CountValidator::sumTrackDifference = 0;

  CountValidator::CountValidator(edm::ProductRegistry& reg)
      : digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
        trackCountToken_(reg.consumes<TrackCount>()),
        vertexCountToken_(reg.consumes<VertexCount>()),
        digiToken_(reg.consumes<SiPixelDigisAlpaka>()),
        clusterToken_(reg.consumes<SiPixelClustersAlpaka>()),
        trackToken_(reg.consumes<PixelTrackHost>()),
        vertexToken_(reg.consumes<ZVertexHost>()) {}

  void CountValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    constexpr float trackTolerance = 0.012f;  // in 200 runs of 1k events all events are withing this tolerance
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    // For some reason the TBB backend occasionally gives difference of 2
    constexpr int vertexTolerance = 2;
#else
    constexpr int vertexTolerance = 1;
#endif
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
      auto const& tracksPtr = iEvent.get(trackToken_);
      auto const tracks = tracksPtr.get();

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
      auto const& verticesPtr = iEvent.get(vertexToken_);
      auto const vertices = verticesPtr.get();

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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CountValidator);
