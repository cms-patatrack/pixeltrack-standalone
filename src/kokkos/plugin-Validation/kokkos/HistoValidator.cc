#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"
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
  class HistoValidator : public edm::EDProducerExternalWork {
  public:
    explicit HistoValidator(edm::ProductRegistry& reg);

  private:
    void acquire(const edm::Event& iEvent,
                 const edm::EventSetup& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endJob() override;

    edm::EDGetTokenT<cms::kokkos::Product<SiPixelDigisKokkos<KokkosExecSpace>>> digiToken_;
    edm::EDGetTokenT<cms::kokkos::Product<SiPixelClustersKokkos<KokkosExecSpace>>> clusterToken_;
    edm::EDGetTokenT<cms::kokkos::Product<TrackingRecHit2DKokkos<KokkosExecSpace>>> hitToken_;
    edm::EDGetTokenT<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>::HostMirror> trackToken_;
    edm::EDGetTokenT<Kokkos::View<ZVertexSoA, KokkosExecSpace>::HostMirror> vertexToken_;

    uint32_t nDigis;
    uint32_t nModules;
    uint32_t nClusters;
    uint32_t nHits;

    Kokkos::View<uint16_t const*, KokkosExecSpace>::HostMirror h_adc;
    Kokkos::View<uint32_t const*, KokkosExecSpace>::HostMirror h_clusInModule;

    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_lx;
    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_ly;
    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_lex;
    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_ley;
    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_gx;
    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_gy;
    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_gz;
    Kokkos::View<float const*, KokkosExecSpace>::HostMirror h_gr;
    Kokkos::View<int32_t const*, KokkosExecSpace>::HostMirror h_charge;
    Kokkos::View<int16_t const*, KokkosExecSpace>::HostMirror h_sizex;
    Kokkos::View<int16_t const*, KokkosExecSpace>::HostMirror h_sizey;

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
      : digiToken_(reg.consumes<cms::kokkos::Product<SiPixelDigisKokkos<KokkosExecSpace>>>()),
        clusterToken_(reg.consumes<cms::kokkos::Product<SiPixelClustersKokkos<KokkosExecSpace>>>()),
        hitToken_(reg.consumes<cms::kokkos::Product<TrackingRecHit2DKokkos<KokkosExecSpace>>>()),
        trackToken_(reg.consumes<Kokkos::View<pixelTrack::TrackSoA, KokkosExecSpace>::HostMirror>()),
        vertexToken_(reg.consumes<Kokkos::View<ZVertexSoA, KokkosExecSpace>::HostMirror>()) {}

  void HistoValidator::acquire(const edm::Event& iEvent,
                               const edm::EventSetup& iSetup,
                               edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& pdigis = iEvent.get(digiToken_);
    cms::kokkos::ScopedContextAcquire<KokkosExecSpace> ctx{pdigis, std::move(waitingTaskHolder)};
    auto const& digis = ctx.get(pdigis);
    auto const& clusters = ctx.get(iEvent, clusterToken_);
    auto const& hits = ctx.get(iEvent, hitToken_);

    nDigis = digis.nDigis();
    nModules = digis.nModules();
    h_adc = digis.adcToHostAsync(ctx.execSpace());

    nClusters = clusters.nClusters();
    auto const d_clusInModule = clusters.clusInModule();
    h_clusInModule = Kokkos::create_mirror_view(d_clusInModule);
    Kokkos::deep_copy(ctx.execSpace(), h_clusInModule, d_clusInModule);

    nHits = hits.nHits();
    h_lx = hits.xlToHostAsync(ctx.execSpace());
    h_ly = hits.ylToHostAsync(ctx.execSpace());
    h_lex = hits.xerrToHostAsync(ctx.execSpace());
    h_ley = hits.yerrToHostAsync(ctx.execSpace());
    h_gx = hits.xgToHostAsync(ctx.execSpace());
    h_gy = hits.ygToHostAsync(ctx.execSpace());
    h_gz = hits.zgToHostAsync(ctx.execSpace());
    h_gr = hits.rgToHostAsync(ctx.execSpace());
    h_charge = hits.chargeToHostAsync(ctx.execSpace());
    h_sizex = hits.xsizeToHostAsync(ctx.execSpace());
    h_sizey = hits.ysizeToHostAsync(ctx.execSpace());
  }

  void HistoValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    histos["digi_n"].fill(nDigis);
    for (uint32_t i = 0; i < nDigis; ++i) {
      histos["digi_adc"].fill(h_adc(i));
    }
    histos["module_n"].fill(nModules);
    //h_adc.reset();

    histos["cluster_n"].fill(nClusters);
    for (uint32_t i = 0; i < nModules; ++i) {
      histos["cluster_per_module_n"].fill(h_clusInModule(i));
    }
    //h_clusInModule.reset();

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
    /*
    h_lx.reset();
    h_ly.reset();
    h_lex.reset();
    h_ley.reset();
    h_gx.reset();
    h_gy.reset();
    h_gz.reset();
    h_gr.reset();
    h_charge.reset();
    h_sizex.reset();
    h_sizey.reset();
    */

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
