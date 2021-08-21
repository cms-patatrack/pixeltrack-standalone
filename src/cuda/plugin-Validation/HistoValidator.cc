#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "SimpleAtomicHisto.h"

#include <map>
#include <fstream>

struct HistoValidator_AsyncState {
  uint32_t nDigis;
  uint32_t nModules;
  uint32_t nClusters;
  uint32_t nHits;
  cms::cuda::host::unique_ptr<uint16_t[]> adc;
  cms::cuda::host::unique_ptr<uint32_t[]> clusInModule;
  cms::cuda::host::unique_ptr<float[]> localCoord;
  cms::cuda::host::unique_ptr<float[]> globalCoord;
  cms::cuda::host::unique_ptr<int32_t[]> charge;
  cms::cuda::host::unique_ptr<int16_t[]> size;
};

class HistoValidator : public edm::EDProducerExternalWork<HistoValidator_AsyncState> {
public:
  explicit HistoValidator(edm::ProductRegistry& reg);

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder,
               AsyncState& state) const override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup, AsyncState& state) override;
  void endJob() override;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiToken_;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterToken_;
  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DCUDA>> hitToken_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> trackToken_;
  edm::EDGetTokenT<ZVertexHeterogeneous> vertexToken_;

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
    : digiToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()),
      hitToken_(reg.consumes<cms::cuda::Product<TrackingRecHit2DCUDA>>()),
      trackToken_(reg.consumes<PixelTrackHeterogeneous>()),
      vertexToken_(reg.consumes<ZVertexHeterogeneous>()) {}

void HistoValidator::acquire(const edm::Event& iEvent,
                             const edm::EventSetup& iSetup,
                             edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                             AsyncState& state) const {
  auto const& pdigis = iEvent.get(digiToken_);
  cms::cuda::ScopedContextAcquire ctx{pdigis, std::move(waitingTaskHolder)};
  auto const& digis = ctx.get(iEvent, digiToken_);
  auto const& clusters = ctx.get(iEvent, clusterToken_);
  auto const& hits = ctx.get(iEvent, hitToken_);

  state.nDigis = digis.nDigis();
  state.nModules = digis.nModules();
  state.adc = digis.adcToHostAsync(ctx.stream());

  state.nClusters = clusters.nClusters();
  state.clusInModule = cms::cuda::make_host_unique<uint32_t[]>(state.nModules, ctx.stream());
  cudaCheck(cudaMemcpyAsync(state.clusInModule.get(),
                            clusters.clusInModule(),
                            sizeof(uint32_t) * state.nModules,
                            cudaMemcpyDefault,
                            ctx.stream()));

  state.nHits = hits.nHits();
  state.localCoord = hits.localCoordToHostAsync(ctx.stream());
  state.globalCoord = hits.globalCoordToHostAsync(ctx.stream());
  state.charge = hits.chargeToHostAsync(ctx.stream());
  state.size = hits.sizeToHostAsync(ctx.stream());
}

void HistoValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup, AsyncState& state) {
  histos["digi_n"].fill(state.nDigis);
  for (uint32_t i = 0; i < state.nDigis; ++i) {
    histos["digi_adc"].fill(state.adc[i]);
  }
  //adc.reset();
  histos["module_n"].fill(state.nModules);

  histos["cluster_n"].fill(state.nClusters);
  for (uint32_t i = 0; i < state.nModules; ++i) {
    histos["cluster_per_module_n"].fill(state.clusInModule[i]);
  }
  //clusInModule.reset();

  histos["hit_n"].fill(state.nHits);
  for (uint32_t i = 0; i < state.nHits; ++i) {
    histos["hit_lx"].fill(state.localCoord[i]);
    histos["hit_ly"].fill(state.localCoord[i + state.nHits]);
    histos["hit_lex"].fill(state.localCoord[i + 2 * state.nHits]);
    histos["hit_ley"].fill(state.localCoord[i + 3 * state.nHits]);
    histos["hit_gx"].fill(state.globalCoord[i]);
    histos["hit_gy"].fill(state.globalCoord[i + state.nHits]);
    histos["hit_gz"].fill(state.globalCoord[i + 2 * state.nHits]);
    histos["hit_gr"].fill(state.globalCoord[i + 3 * state.nHits]);
    histos["hit_charge"].fill(state.charge[i]);
    histos["hit_sizex"].fill(state.size[i]);
    histos["hit_sizey"].fill(state.size[i + state.nHits]);
  }
  //state.localCoord.reset();
  //state.globalCoord.reset();
  //state.charge.reset();
  //state.size.reset();

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
  std::ofstream out("histograms_cuda.txt");
  for (auto const& elem : histos) {
    out << elem.first << " " << elem.second << "\n";
  }
}

DEFINE_FWK_MODULE(HistoValidator);
