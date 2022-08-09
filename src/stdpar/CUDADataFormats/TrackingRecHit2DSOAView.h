#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include "CUDADataFormats/gpuClusteringConstants.h"
#include "CUDACore/HistoContainer.h"
#include "Geometry/phase1PixelTopology.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint16_t;  // if above is <=2^16

  using Hist =
      cms::cuda::HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), uint16_t, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  friend class TrackingRecHit2D;

  __forceinline__ uint32_t nHits() const { return m_nHits; }

  __forceinline__ float& xLocal(int i) { return m_xl[i]; }
  __forceinline__ float xLocal(int i) const { return m_xl[i]; }
  __forceinline__ float& yLocal(int i) { return m_yl[i]; }
  __forceinline__ float yLocal(int i) const { return m_yl[i]; }

  __forceinline__ float& xerrLocal(int i) { return m_xerr[i]; }
  __forceinline__ float xerrLocal(int i) const { return m_xerr[i]; }
  __forceinline__ float& yerrLocal(int i) { return m_yerr[i]; }
  __forceinline__ float yerrLocal(int i) const { return m_yerr[i]; }

  __forceinline__ float& xGlobal(int i) { return m_xg[i]; }
  __forceinline__ float xGlobal(int i) const { return m_xg[i]; }
  __forceinline__ float& yGlobal(int i) { return m_yg[i]; }
  __forceinline__ float yGlobal(int i) const { return m_yg[i]; }
  __forceinline__ float& zGlobal(int i) { return m_zg[i]; }
  __forceinline__ float zGlobal(int i) const { return m_zg[i]; }
  __forceinline__ float& rGlobal(int i) { return m_rg[i]; }
  __forceinline__ float rGlobal(int i) const { return m_rg[i]; }

  __forceinline__ int16_t& iphi(int i) { return m_iphi[i]; }
  __forceinline__ int16_t iphi(int i) const { return m_iphi[i]; }

  __forceinline__ int32_t& charge(int i) { return m_charge[i]; }
  __forceinline__ int32_t charge(int i) const { return m_charge[i]; }
  __forceinline__ int16_t& clusterSizeX(int i) { return m_xsize[i]; }
  __forceinline__ int16_t clusterSizeX(int i) const { return m_xsize[i]; }
  __forceinline__ int16_t& clusterSizeY(int i) { return m_ysize[i]; }
  __forceinline__ int16_t clusterSizeY(int i) const { return m_ysize[i]; }
  __forceinline__ uint16_t& detectorIndex(int i) { return m_detInd[i]; }
  __forceinline__ uint16_t detectorIndex(int i) const { return m_detInd[i]; }

  __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  __forceinline__ uint32_t hitsModuleStart(int i) const { return m_hitsModuleStart[i]; }

  __forceinline__ uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  __forceinline__ uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  __forceinline__ Hist& phiBinner() { return *m_hist; }
  __forceinline__ Hist const& phiBinner() const { return *m_hist; }

  __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // local coord
  float *m_xl, *m_yl;
  float *m_xerr, *m_yerr;

  // global coord
  float *m_xg, *m_yg, *m_zg, *m_rg;
  int16_t* m_iphi;

  // cluster properties
  int32_t* m_charge;
  int16_t* m_xsize;
  int16_t* m_ysize;
  uint16_t* m_detInd;

  // supporting objects
  AverageGeometry* m_averageGeometry;  // owned (corrected for beam spot: not sure where to host it otherwise)
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  Hist* m_hist;

  uint32_t m_nHits;
};

#endif
