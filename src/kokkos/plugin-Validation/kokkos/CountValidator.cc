#include "KokkosCore/kokkosConfig.h"
#include "KokkosDataFormats/PixelTrackKokkos.h"
#include "KokkosDataFormats/SiPixelClustersKokkos.h"
#include "KokkosDataFormats/SiPixelDigisKokkos.h"
#include "DataFormats/ZVertexSoA.h"
#include "DataFormats/DigiClusterCount.h"
#include "DataFormats/TrackCount.h"
#include "DataFormats/VertexCount.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include <atomic>
#include <iostream>
#include <sstream>

namespace {
  std::atomic<int> allEvents{0};
  std::atomic<int> goodEvents{0};
}  // namespace

namespace KOKKOS_NAMESPACE {

  class CountValidator : public edm::EDProducer {
  public:
    explicit CountValidator(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endJob() override;

    edm::EDGetTokenT<DigiClusterCount> digiClusterCountToken_;
    edm::EDGetTokenT<TrackCount> trackCountToken_;
    edm::EDGetTokenT<VertexCount> vertexCountToken_;

    edm::EDGetTokenT<SiPixelDigisKokkos<KokkosExecSpace>> digiToken_;
    edm::EDGetTokenT<SiPixelClustersKokkos<KokkosExecSpace>> clusterToken_;
    edm::EDGetTokenT<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>::HostMirror> trackToken_;
    edm::EDGetTokenT<Kokkos::View<ZVertexSoA, KokkosExecSpace>::HostMirror> vertexToken_;
  };

  CountValidator::CountValidator(edm::ProductRegistry& reg)
      : digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
        trackCountToken_(reg.consumes<TrackCount>()),
        vertexCountToken_(reg.consumes<VertexCount>()),
        digiToken_(reg.consumes<SiPixelDigisKokkos<KokkosExecSpace>>()),
        clusterToken_(reg.consumes<SiPixelClustersKokkos<KokkosExecSpace>>()),
        trackToken_(reg.consumes<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>::HostMirror>()),
        vertexToken_(reg.consumes<Kokkos::View<ZVertexSoA, KokkosExecSpace>::HostMirror>()) {}

  void CountValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // values from cuda program
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
      for (int i = 0; i < tracks().stride(); ++i) {
        if (tracks().nHits(i) > 0) {
          ++nTracks;
        }
      }

      auto rel = std::abs(float(nTracks - int(count.nTracks())) / count.nTracks());
      if (rel >= trackTolerance) {
        ss << "\n N(tracks) is " << nTracks << " expected " << count.nTracks() << ", relative difference " << rel
           << " is outside tolerance " << trackTolerance;
        ok = false;
      }
    }

    {
      auto const& count = iEvent.get(vertexCountToken_);
      auto const& vertices = iEvent.get(vertexToken_);

      auto diff = std::abs(int(vertices().nvFinal) - int(count.nVertices()));
      if (diff > vertexTolerance) {
        ss << "\n N(vertices) is " << vertices().nvFinal << " expected " << count.nVertices() << ", difference " << diff
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
    } else {
      std::cout << "CountValidator: " << (allEvents - goodEvents) << " events failed validation (see details above)\n";
      throw std::runtime_error("CountValidator failed");
    }
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(CountValidator);
