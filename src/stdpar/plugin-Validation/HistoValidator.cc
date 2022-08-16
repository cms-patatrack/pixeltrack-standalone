#include "CUDADataFormats/PixelTrack.h"
#include "CUDADataFormats/SiPixelClusters.h"
#include "CUDADataFormats/SiPixelDigis.h"
#include "CUDADataFormats/TrackingRecHit2D.h"
#include "CUDADataFormats/ZVertex.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "SimpleAtomicHisto.h"

#include <map>
#include <fstream>

class HistoValidator : public edm::EDProducer {
public:
  explicit HistoValidator(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void endJob() override;

  edm::EDGetTokenT<SiPixelDigis> digiToken_;
  edm::EDGetTokenT<SiPixelClusters> clusterToken_;
  edm::EDGetTokenT<TrackingRecHit2D> hitToken_;
  edm::EDGetTokenT<PixelTrack> trackToken_;
  edm::EDGetTokenT<ZVertex> vertexToken_;

  static std::map<std::string, SimpleAtomicHisto> histos;
};

std::map<std::string, SimpleAtomicHisto> HistoValidator::histos = {
    {"digi_n", SimpleAtomicHisto(100, 0, 1e5)},
    {"digi_adc", SimpleAtomicHisto(250, 0, 5e4)},
    {"module_n", SimpleAtomicHisto(100, 1500, 2000)},
    {"cluster_n", SimpleAtomicHisto(200, 5000, 25000)},
    {"cluster_per_module_n", SimpleAtomicHisto(110, 0, 110)},
    {"hit_n", SimpleAtomicHisto(200, 5000, 25000)},
    {"hit_lx", SimpleAtomicHisto(200, -1, 1)},
    {"hit_ly", SimpleAtomicHisto(800, -4, 4)},
    {"hit_lex", SimpleAtomicHisto(100, 0, 5e-5)},
    {"hit_ley", SimpleAtomicHisto(100, 0, 1e-4)},
    {"hit_gx", SimpleAtomicHisto(200, -20, 20)},
    {"hit_gy", SimpleAtomicHisto(200, -20, 20)},
    {"hit_gz", SimpleAtomicHisto(600, -60, 60)},
    {"hit_gr", SimpleAtomicHisto(200, 0, 20)},
    {"hit_charge", SimpleAtomicHisto(400, 0, 4e6)},
    {"hit_sizex", SimpleAtomicHisto(800, 0, 800)},
    {"hit_sizey", SimpleAtomicHisto(800, 0, 800)},
    {"track_n", SimpleAtomicHisto(150, 0, 15000)},
    {"track_nhits", SimpleAtomicHisto(3, 3, 6)},
    {"track_chi2", SimpleAtomicHisto(100, 0, 40)},
    {"track_pt", SimpleAtomicHisto(400, 0, 400)},
    {"track_eta", SimpleAtomicHisto(100, -3, 3)},
    {"track_phi", SimpleAtomicHisto(100, -3.15, 3.15)},
    {"track_tip", SimpleAtomicHisto(100, -1, 1)},
    {"track_tip_zoom", SimpleAtomicHisto(100, -0.05, 0.05)},
    {"track_zip", SimpleAtomicHisto(100, -15, 15)},
    {"track_zip_zoom", SimpleAtomicHisto(100, -0.1, 0.1)},
    {"track_quality", SimpleAtomicHisto(6, 0, 6)},
    {"vertex_n", SimpleAtomicHisto(60, 0, 60)},
    {"vertex_z", SimpleAtomicHisto(100, -15, 15)},
    {"vertex_chi2", SimpleAtomicHisto(100, 0, 40)},
    {"vertex_ndof", SimpleAtomicHisto(170, 0, 170)},
    {"vertex_pt2", SimpleAtomicHisto(100, 0, 4000)}};

HistoValidator::HistoValidator(edm::ProductRegistry& reg)
    : digiToken_(reg.consumes<SiPixelDigis>()),
      clusterToken_(reg.consumes<SiPixelClusters>()),
      hitToken_(reg.consumes<TrackingRecHit2D>()),
      trackToken_(reg.consumes<PixelTrack>()),
      vertexToken_(reg.consumes<ZVertex>()) {}

void HistoValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& digis = iEvent.get(digiToken_);
  auto const& clusters = iEvent.get(clusterToken_);
  auto const& hits = iEvent.get(hitToken_);

  uint32_t nDigis = digis.nDigis();
  uint32_t nModules = digis.nModules();
  uint16_t const* h_adc = digis.adc();

  uint32_t nClusters = clusters.nClusters();
  uint32_t const* h_clusInModule = clusters.clusInModule();

  uint32_t nHits = hits.nHits();

  float const* h_localCoord = hits.localCoord();
  float const* h_globalCoord = hits.globalCoord();
  int32_t const* h_charge = hits.charge();
  int16_t const* h_size = hits.size();

  histos["digi_n"].fill(nDigis);
  for (uint32_t i = 0; i < nDigis; ++i) {
    histos["digi_adc"].fill(h_adc[i]);
  }
  histos["module_n"].fill(nModules);

  histos["cluster_n"].fill(nClusters);
  for (uint32_t i = 0; i < nModules; ++i) {
    histos["cluster_per_module_n"].fill(h_clusInModule[i]);
  }

  histos["hit_n"].fill(nHits);
  for (uint32_t i = 0; i < nHits; ++i) {
    histos["hit_lx"].fill(h_localCoord[i]);
    histos["hit_ly"].fill(h_localCoord[i + nHits]);
    histos["hit_lex"].fill(h_localCoord[i + 2 * nHits]);
    histos["hit_ley"].fill(h_localCoord[i + 3 * nHits]);
    histos["hit_gx"].fill(h_globalCoord[i]);
    histos["hit_gy"].fill(h_globalCoord[i + nHits]);
    histos["hit_gz"].fill(h_globalCoord[i + 2 * nHits]);
    histos["hit_gr"].fill(h_globalCoord[i + 3 * nHits]);
    histos["hit_charge"].fill(h_charge[i]);
    histos["hit_sizex"].fill(h_size[i]);
    histos["hit_sizey"].fill(h_size[i + nHits]);
  }

  {
    auto const& tracks = iEvent.get(trackToken_);

    int nTracks = 0;
    for (int i = 0; i < tracks->stride(); ++i) {
      if (tracks->nHits(i) > 0 and tracks->quality(i) >= trackQuality::loose) {
        ++nTracks;
        histos["track_nhits"].fill(tracks->nHits(i));
        histos["track_chi2"].fill(tracks->chi2(i));
        histos["track_pt"].fill(tracks->pt(i));
        histos["track_eta"].fill(tracks->eta(i));
        histos["track_phi"].fill(tracks->phi(i));
        histos["track_tip"].fill(tracks->tip(i));
        histos["track_tip_zoom"].fill(tracks->tip(i));
        histos["track_zip"].fill(tracks->zip(i));
        histos["track_zip_zoom"].fill(tracks->zip(i));
        histos["track_quality"].fill(tracks->quality(i));
      }
    }

    histos["track_n"].fill(nTracks);
  }

  {
    auto const& vertices = iEvent.get(vertexToken_);

    histos["vertex_n"].fill(vertices->nvFinal);
    for (uint32_t i = 0; i < vertices->nvFinal; ++i) {
      histos["vertex_z"].fill(vertices->zv[i]);
      histos["vertex_chi2"].fill(vertices->chi2[i]);
      histos["vertex_ndof"].fill(vertices->ndof[i]);
      histos["vertex_pt2"].fill(vertices->ptv2[i]);
    }
  }
}

void HistoValidator::endJob() {
  std::ofstream out("histograms_cudauvm.txt");
  for (auto const& elem : histos) {
    out << elem.first << " " << elem.second << "\n";
  }
}

DEFINE_FWK_MODULE(HistoValidator);
