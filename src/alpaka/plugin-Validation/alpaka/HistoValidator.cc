#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"
//#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/SiPixelClustersAlpaka.h"
#include "AlpakaDataFormats/SiPixelDigisAlpaka.h"
//#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"
//#include "DataFormats/ZVertexSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "../SimpleAtomicHisto.h"

#include <map>
#include <fstream>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

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

    edm::EDGetTokenT<SiPixelDigisAlpaka> digiToken_;
    edm::EDGetTokenT<SiPixelClustersAlpaka> clusterToken_;
    //edm::EDGetTokenT<TrackingRecHit2DAlpaka> hitToken_;
    //edm::EDGetTokenT<PixelTrackHeterogeneous> trackToken_;
    //edm::EDGetTokenT<ZVertexHeterogeneous> vertexToken_;

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
      : digiToken_(reg.consumes<SiPixelDigisAlpaka>()),
        clusterToken_(reg.consumes<SiPixelClustersAlpaka>())  //,
                                                              //hitToken_(reg.consumes<TrackingRecHit2DAlpaka>()),
                                                              //trackToken_(reg.consumes<PixelTrackHeterogeneous>()),
                                                              //vertexToken_(reg.consumes<ZVertexHeterogeneous>())
  {}

#ifdef TODO
  void HistoValidator::acquire(const edm::Event& iEvent,
                               const edm::EventSetup& iSetup,
                               edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& pdigis = iEvent.get(digiToken_);
    cms::cuda::ScopedContextAcquire ctx{pdigis, std::move(waitingTaskHolder)};
    auto const& digis = ctx.get(iEvent, digiToken_);
    auto const& clusters = ctx.get(iEvent, clusterToken_);
    auto const& hits = ctx.get(iEvent, hitToken_);

    nDigis = digis.nDigis();
    nModules = digis.nModules();
    h_adc = digis.adcToHostAsync(ctx.stream());

    nClusters = clusters.nClusters();
    h_clusInModule = cms::cuda::make_host_unique<uint32_t[]>(nModules, ctx.stream());
    cudaCheck(cudaMemcpyAsync(
        h_clusInModule.get(), clusters.clusInModule(), sizeof(uint32_t) * nModules, cudaMemcpyDefault, ctx.stream()));

    nHits = hits.nHits();
    h_localCoord = hits.localCoordToHostAsync(ctx.stream());
    h_globalCoord = hits.globalCoordToHostAsync(ctx.stream());
    h_charge = hits.chargeToHostAsync(ctx.stream());
    h_size = hits.sizeToHostAsync(ctx.stream());
  }
#endif

  void HistoValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& digis = iEvent.get(digiToken_);
    auto const& clusters = iEvent.get(clusterToken_);
    //auto const& hits = iEvent.get(hitToken_);

    auto const nDigis = digis.nDigis();
    auto const nModules = digis.nModules();
    auto const h_adcBuf = digis.adcToHost();
    auto const h_adc = alpaka::getPtrNative(h_adcBuf);

    auto const nClusters = clusters.nClusters();
    auto const d_clusInModuleView =
        cms::alpakatools::createDeviceView<uint32_t>(clusters.clusInModule(), gpuClustering::MaxNumModules);
    auto h_clusInModuleBuf{cms::alpakatools::allocHostBuf<uint32_t>(gpuClustering::MaxNumModules)};
    Queue queue(device);
    alpaka::memcpy(queue, h_clusInModuleBuf, d_clusInModuleView, gpuClustering::MaxNumModules);
    auto h_clusInModule = alpaka::getPtrNative(h_clusInModuleBuf);

    /*
  auto const nHits = hits.nHits();
  auto const h_lx = hits.xlToHostAsync();
  auto const h_ly = hits.ylToHostAsync();
  auto const h_lex = hits.xerrToHostAsync();
  auto const h_ley = hits.yerrToHostAsync();
  auto const h_gx = hits.xgToHostAsync();
  auto const h_gy = hits.ygToHostAsync();
  auto const h_gz = hits.zgToHostAsync();
  auto const h_gr = hits.rgToHostAsync();
  auto const h_charge = hits.chargeToHostAsync();
  auto const h_sizex = hits.xsizeToHostAsync();
  auto const h_sizey = hits.ysizeToHostAsync();*/

    alpaka::wait(queue);

    histos["digi_n"].fill(nDigis);
    for (uint32_t i = 0; i < nDigis; ++i) {
      histos["digi_adc"].fill(h_adc[i]);
    }
    histos["module_n"].fill(nModules);

    histos["cluster_n"].fill(nClusters);
    for (uint32_t i = 0; i < nModules; ++i) {
      histos["cluster_per_module_n"].fill(h_clusInModule[i]);
    }

    /*
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
  h_localCoord.reset();
  h_globalCoord.reset();
  h_charge.reset();
  h_size.reset();

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
    }*/
  }

  void HistoValidator::endJob() {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    std::ofstream out("histograms_alpaka_serial.txt");
#elif defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    std::string fname =
        "histograms_alpaka_tbb_.txt";  // TO DO: access number of threads from TBB pool from Alpaka TBB backend and add to file name
    std::ofstream out(fname.c_str());
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
