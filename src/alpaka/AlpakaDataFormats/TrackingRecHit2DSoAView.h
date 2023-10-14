#ifndef AlpakaDataFormats_TrackingRecHit2DSoAView_h
#define AlpakaDataFormats_TrackingRecHit2DSoAView_h

#include "AlpakaDataFormats/gpuClusteringConstants.h"
#include "AlpakaCore/HistoContainer.h"
#include "Geometry/phase1PixelTopology.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

namespace alpaka_cuda_async {
  class TrackingRecHit2DAlpaka;
}

namespace alpaka_rocm_async {
  class TrackingRecHit2DAlpaka;
}

namespace alpaka_tbb_async {
  class TrackingRecHit2DAlpaka;
}

namespace alpaka_serial_sync {
  class TrackingRecHit2DAlpaka;
}

namespace alpaka_cpu_sycl {
  class TrackingRecHit2DAlpaka;
}

namespace alpaka_gpu_sycl {
  class TrackingRecHit2DAlpaka;
}

class TrackingRecHit2DSoAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint16_t;  // if above is <=2^16

  using Hist =
      cms::alpakatools::HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), uint16_t, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  friend class alpaka_cuda_async::TrackingRecHit2DAlpaka;
  friend class alpaka_rocm_async::TrackingRecHit2DAlpaka;
  friend class alpaka_tbb_async::TrackingRecHit2DAlpaka;
  friend class alpaka_serial_sync::TrackingRecHit2DAlpaka;
  friend class alpaka_cpu_sycl::TrackingRecHit2DAlpaka;
  friend class alpaka_gpu_sycl::TrackingRecHit2DAlpaka;

  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t nHits() const { return m_nHits; }

  // TO DO: removed __ldg, check impact on perf
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& xLocal(int i) { return m_xl[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float xLocal(int i) const { return m_xl[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& yLocal(int i) { return m_yl[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float yLocal(int i) const { return m_yl[i]; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& xerrLocal(int i) { return m_xerr[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float xerrLocal(int i) const { return m_xerr[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& yerrLocal(int i) { return m_yerr[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float yerrLocal(int i) const { return m_yerr[i]; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& xGlobal(int i) { return m_xg[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float xGlobal(int i) const { return m_xg[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& yGlobal(int i) { return m_yg[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float yGlobal(int i) const { return m_yg[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& zGlobal(int i) { return m_zg[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float zGlobal(int i) const { return m_zg[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float& rGlobal(int i) { return m_rg[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float rGlobal(int i) const { return m_rg[i]; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t& iphi(int i) { return m_iphi[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t iphi(int i) const { return m_iphi[i]; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t& charge(int i) { return m_charge[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t charge(int i) const { return m_charge[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t& clusterSizeX(int i) { return m_xsize[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t clusterSizeX(int i) const { return m_xsize[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t& clusterSizeY(int i) { return m_ysize[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t clusterSizeY(int i) const { return m_ysize[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint16_t& detectorIndex(int i) { return m_detInd[i]; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint16_t detectorIndex(int i) const { return m_detInd[i]; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t hitsModuleStart(int i) const { return m_hitsModuleStart[i]; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE Hist& phiBinner() { return *m_hist; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Hist const& phiBinner() const { return *m_hist; }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // TO DO: NB: added __restrict__ versus legacy

  // local coord
  float* __restrict__ m_xl;
  float* __restrict__ m_yl;
  float* __restrict__ m_xerr;
  float* __restrict__ m_yerr;

  // global coord
  float* __restrict__ m_xg;
  float* __restrict__ m_yg;
  float* __restrict__ m_zg;
  float* __restrict__ m_rg;
  int16_t* __restrict__ m_iphi;

  // cluster properties
  int32_t* __restrict__ m_charge;
  int16_t* __restrict__ m_xsize;
  int16_t* __restrict__ m_ysize;
  uint16_t* __restrict__ m_detInd;

  // supporting objects
  AverageGeometry* m_averageGeometry;  // owned (corrected for beam spot: not sure where to host it otherwise)
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  Hist* m_hist;

  uint32_t m_nHits;
};

#endif  // AlpakaDataFormats_TrackingRecHit2DSoAView_h
