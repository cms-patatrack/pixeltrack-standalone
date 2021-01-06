#include "KokkosDataFormats/PixelTrackKokkos.h"
#include "KokkosDataFormats/SiPixelClustersKokkos.h"
#include "KokkosDataFormats/SiPixelDigisKokkos.h"
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"
#include "DataFormats/ZVertexSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "../SimpleAtomicHisto.h"

#include <map>
#include <fstream>

namespace KOKKOS_NAMESPACE {
  class HistoValidator : public edm::EDProducer {
  public:
    explicit HistoValidator(edm::ProductRegistry& reg);

  private:
#ifdef TODO
    void acquire(const edm::Event& iEvent,
                 const edm::EventSetup& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
#endif
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endJob() override;

    edm::EDGetTokenT<SiPixelDigisKokkos<KokkosExecSpace>> digiToken_;
    edm::EDGetTokenT<SiPixelClustersKokkos<KokkosExecSpace>> clusterToken_;
    edm::EDGetTokenT<TrackingRecHit2DKokkos<KokkosExecSpace>> hitToken_;
    edm::EDGetTokenT<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>::HostMirror> trackToken_;
    edm::EDGetTokenT<Kokkos::View<ZVertexSoA, KokkosExecSpace>::HostMirror> vertexToken_;

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
      : digiToken_(reg.consumes<SiPixelDigisKokkos<KokkosExecSpace>>()),
        clusterToken_(reg.consumes<SiPixelClustersKokkos<KokkosExecSpace>>()),
        hitToken_(reg.consumes<TrackingRecHit2DKokkos<KokkosExecSpace>>()),
        trackToken_(reg.consumes<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>::HostMirror>()),
        vertexToken_(reg.consumes<Kokkos::View<ZVertexSoA, KokkosExecSpace>::HostMirror>()) {}

  void HistoValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& digis = iEvent.get(digiToken_);
    auto const& clusters = iEvent.get(clusterToken_);
    auto const& hits = iEvent.get(hitToken_);

    auto const nDigis = digis.nDigis();
    auto const nModules = digis.nModules();
    auto const h_adc = digis.adcToHostAsync(KokkosExecSpace());

    auto const nClusters = clusters.nClusters();
    auto const d_clusInModule = clusters.clusInModule();
    auto h_clusInModule = Kokkos::create_mirror_view(d_clusInModule);
    Kokkos::deep_copy(KokkosExecSpace(), h_clusInModule, d_clusInModule);

    auto const nHits = hits.nHits();
    auto const h_lx = hits.xlToHostAsync(KokkosExecSpace());
    auto const h_ly = hits.ylToHostAsync(KokkosExecSpace());
    auto const h_lex = hits.xerrToHostAsync(KokkosExecSpace());
    auto const h_ley = hits.yerrToHostAsync(KokkosExecSpace());
    auto const h_gx = hits.xgToHostAsync(KokkosExecSpace());
    auto const h_gy = hits.ygToHostAsync(KokkosExecSpace());
    auto const h_gz = hits.zgToHostAsync(KokkosExecSpace());
    auto const h_gr = hits.rgToHostAsync(KokkosExecSpace());
    auto const h_charge = hits.chargeToHostAsync(KokkosExecSpace());
    auto const h_sizex = hits.xsizeToHostAsync(KokkosExecSpace());
    auto const h_sizey = hits.ysizeToHostAsync(KokkosExecSpace());

    KokkosExecSpace().fence();

    histos["digi_n"].fill(nDigis);
    for (uint32_t i = 0; i < nDigis; ++i) {
      histos["digi_adc"].fill(h_adc(i));
    }
    histos["module_n"].fill(nModules);

    histos["cluster_n"].fill(nClusters);
    for (uint32_t i = 0; i < nModules; ++i) {
      histos["cluster_per_module_n"].fill(h_clusInModule(i));
    }

    histos["hit_n"].fill(nHits);
    for (uint32_t i = 0; i < nHits; ++i) {
      histos["hit_lx"].fill(h_lx(i));
      histos["hit_ly"].fill(h_ly(i));
      histos["hit_lex"].fill(h_lex(i));
      histos["hit_ley"].fill(h_ley(i));
      histos["hit_gx"].fill(h_gx(i));
      histos["hit_gy"].fill(h_gy(i));
      histos["hit_gz"].fill(h_gz(i));
      histos["hit_gr"].fill(h_gr(i));
      histos["hit_charge"].fill(h_charge(i));
      histos["hit_sizex"].fill(h_sizex(i));
      histos["hit_sizey"].fill(h_sizey(i));
    }

    {
      auto const& tracks = iEvent.get(trackToken_);

      int nTracks = 0;
      for (int i = 0; i < tracks().stride(); ++i) {
        if (tracks().nHits(i) > 0 and tracks().quality(i) >= trackQuality::loose) {
          ++nTracks;
          histos["track_nhits"].fill(tracks().nHits(i));
          histos["track_chi2"].fill(tracks().chi2(i));
          histos["track_pt"].fill(tracks().pt(i));
          histos["track_eta"].fill(tracks().eta(i));
          histos["track_phi"].fill(tracks().phi(i));
          histos["track_tip"].fill(tracks().tip(i));
          histos["track_tip_zoom"].fill(tracks().tip(i));
          histos["track_zip"].fill(tracks().zip(i));
          histos["track_zip_zoom"].fill(tracks().zip(i));
          histos["track_quality"].fill(tracks().quality(i));
        }
      }

      histos["track_n"].fill(nTracks);
    }

    {
      auto const& vertices = iEvent.get(vertexToken_);

      histos["vertex_n"].fill(vertices().nvFinal);
      for (uint32_t i = 0; i < vertices().nvFinal; ++i) {
        histos["vertex_z"].fill(vertices().zv[i]);
        histos["vertex_chi2"].fill(vertices().chi2[i]);
        histos["vertex_ndof"].fill(vertices().ndof[i]);
        histos["vertex_pt2"].fill(vertices().ptv2[i]);
      }
    }
  }

  void HistoValidator::endJob() {
#ifdef KOKKOS_BACKEND_SERIAL
    std::ofstream out("histograms_kokkos_serial.txt");
#elif defined KOKKOS_BACKEND_PTHREAD
    std::string fname =
        "histograms_kokkos_pthread_" + std::to_string(KokkosExecSpace::impl_thread_pool_size()) + ".txt";
    std::ofstream out(fname.c_str());
#elif defined KOKKOS_BACKEND_CUDA
    std::ofstream out("histograms_kokkos_cuda.txt");
#elif defined KOKKOS_BACKEND_HIP
    std::ofstream out("histograms_kokkos_hip.txt");
#else
#error "Support for a new Kokkos backend must be added here"
#endif
    for (auto const& elem : histos) {
      out << elem.first << " " << elem.second << "\n";
    }
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(HistoValidator);
