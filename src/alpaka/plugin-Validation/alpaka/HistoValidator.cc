#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/SiPixelClustersAlpaka.h"
#include "AlpakaDataFormats/SiPixelDigisAlpaka.h"
#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"
#include "AlpakaDataFormats/ZVertexAlpaka.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "../SimpleAtomicHisto.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HistoValidator : public edm::EDProducerExternalWork {
  public:
    explicit HistoValidator(edm::ProductRegistry& reg);

  private:
    void acquire(const edm::Event& iEvent,
                 const edm::EventSetup& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endJob() override;

    edm::EDGetTokenT<::cms::alpakatools::Product<Queue, SiPixelDigisAlpaka>> digiToken_;
    edm::EDGetTokenT<::cms::alpakatools::Product<Queue, SiPixelClustersAlpaka>> clusterToken_;
    edm::EDGetTokenT<::cms::alpakatools::Product<Queue, TrackingRecHit2DAlpaka>> hitToken_;
    edm::EDGetTokenT<PixelTrackHost> trackToken_;
    edm::EDGetTokenT<ZVertexHost> vertexToken_;

    uint32_t nDigis_;
    uint32_t nModules_;
    uint32_t nClusters_;
    uint32_t nHits_;

    std::optional<AlpakaHostBuf<uint16_t>> h_adc;
    std::optional<AlpakaHostBuf<uint32_t>> h_clusInModule;
    std::optional<AlpakaHostBuf<float>> h_lx;
    std::optional<AlpakaHostBuf<float>> h_ly;
    std::optional<AlpakaHostBuf<float>> h_lex;
    std::optional<AlpakaHostBuf<float>> h_ley;
    std::optional<AlpakaHostBuf<float>> h_gx;
    std::optional<AlpakaHostBuf<float>> h_gy;
    std::optional<AlpakaHostBuf<float>> h_gz;
    std::optional<AlpakaHostBuf<float>> h_gr;
    std::optional<AlpakaHostBuf<int32_t>> h_charge;
    std::optional<AlpakaHostBuf<int16_t>> h_sizex;
    std::optional<AlpakaHostBuf<int16_t>> h_sizey;

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
      : digiToken_{reg.consumes<::cms::alpakatools::Product<Queue, SiPixelDigisAlpaka>>()},
        clusterToken_{reg.consumes<::cms::alpakatools::Product<Queue, SiPixelClustersAlpaka>>()},
        hitToken_{reg.consumes<::cms::alpakatools::Product<Queue, TrackingRecHit2DAlpaka>>()},
        trackToken_{reg.consumes<PixelTrackHost>()},
        vertexToken_{reg.consumes<ZVertexHost>()} {}

  void HistoValidator::acquire(const edm::Event& iEvent,
                               const edm::EventSetup& iSetup,
                               edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& pdigis = iEvent.get(digiToken_);
    ::cms::alpakatools::ScopedContextAcquire ctx{pdigis, std::move(waitingTaskHolder)};
    auto const& digis = ctx.get(pdigis);
    auto const& clusters = ctx.get(iEvent, clusterToken_);
    auto const& hits = ctx.get(iEvent, hitToken_);

    nDigis_ = digis.nDigis();
    nModules_ = digis.nModules();
    h_adc = std::move(digis.adcToHostAsync(ctx.stream()));

    nClusters_ = clusters.nClusters();
    h_clusInModule = ::cms::alpakatools::allocHostBuf<uint32_t>(nModules_);
    alpaka::memcpy(ctx.stream(),
                   *h_clusInModule,
                   ::cms::alpakatools::createDeviceView(ctx.device(), clusters.clusInModule(), nModules_),
                   nModules_);

    nHits_ = hits.nHits();

    h_lx = std::move(hits.xlToHostAsync(ctx.stream()));
    h_ly = std::move(hits.ylToHostAsync(ctx.stream()));
    h_lex = std::move(hits.xerrToHostAsync(ctx.stream()));
    h_ley = std::move(hits.yerrToHostAsync(ctx.stream()));
    h_gx = std::move(hits.xgToHostAsync(ctx.stream()));
    h_gy = std::move(hits.ygToHostAsync(ctx.stream()));
    h_gz = std::move(hits.zgToHostAsync(ctx.stream()));
    h_gr = std::move(hits.rgToHostAsync(ctx.stream()));
    h_charge = std::move(hits.chargeToHostAsync(ctx.stream()));
    h_sizex = std::move(hits.xsizeToHostAsync(ctx.stream()));
    h_sizey = std::move(hits.ysizeToHostAsync(ctx.stream()));
  }

  void HistoValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    histos["module_n"].fill(nModules_);
    histos["digi_n"].fill(nDigis_);
    for (uint32_t i = 0; i < nDigis_; ++i) {
      histos["digi_adc"].fill(alpaka::getPtrNative(*h_adc)[i]);
    }
    h_adc.reset();

    histos["cluster_n"].fill(nClusters_);
    for (uint32_t i = 0; i < nModules_; ++i) {
      histos["cluster_per_module_n"].fill(alpaka::getPtrNative(*h_clusInModule)[i]);
    }
    h_clusInModule.reset();

    histos["hit_n"].fill(nHits_);
    for (uint32_t i = 0; i < nHits_; ++i) {
      histos["hit_lx"].fill(alpaka::getPtrNative(*h_lx)[i]);
      histos["hit_ly"].fill(alpaka::getPtrNative(*h_ly)[i]);
      histos["hit_lex"].fill(alpaka::getPtrNative(*h_lex)[i]);
      histos["hit_ley"].fill(alpaka::getPtrNative(*h_ley)[i]);
      histos["hit_gx"].fill(alpaka::getPtrNative(*h_gx)[i]);
      histos["hit_gy"].fill(alpaka::getPtrNative(*h_gy)[i]);
      histos["hit_gz"].fill(alpaka::getPtrNative(*h_gz)[i]);
      histos["hit_gr"].fill(alpaka::getPtrNative(*h_gr)[i]);
      histos["hit_charge"].fill(alpaka::getPtrNative(*h_charge)[i]);
      histos["hit_sizex"].fill(alpaka::getPtrNative(*h_sizex)[i]);
      histos["hit_sizey"].fill(alpaka::getPtrNative(*h_sizey)[i]);
    }
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

    {
      auto const& tracksBuf = iEvent.get(trackToken_);
      auto const tracks = alpaka::getPtrNative(tracksBuf);

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
      auto const& verticesBuf = iEvent.get(vertexToken_);
      auto const vertices = alpaka::getPtrNative(verticesBuf);

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
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    std::ofstream out("histograms_alpaka_serial.txt");
#elif defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    std::ofstream out("histograms_alpaka_tbb.txt");
#elif defined ALPAKA_ACC_GPU_CUDA_ENABLED
    std::ofstream out("histograms_alpaka_cuda.txt");
#else
#error "Support for a new Alpaka backend must be added here"
#endif
    for (auto const& elem : histos) {
      out << elem.first << " " << elem.second << "\n";
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(HistoValidator);
