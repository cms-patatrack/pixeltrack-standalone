#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include <Kokkos_Core.hpp>

#include "KokkosDataFormats/gpuClusteringConstants.h"
#include "KokkosCore/HistoContainer.h"
#include "Geometry/phase1PixelTopology.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint16_t;  // if above is <=2^16

  using Hist =
      cms::kokkos::HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), uint16_t, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DKokkos;

  KOKKOS_INLINE_FUNCTION uint32_t nHits() const { return m_nHits; }

  KOKKOS_INLINE_FUNCTION float& xLocal(int i) { return m_xl[i]; }
  KOKKOS_INLINE_FUNCTION float xLocal(int i) const { return m_xl[i]; }
  KOKKOS_INLINE_FUNCTION float& yLocal(int i) { return m_yl[i]; }
  KOKKOS_INLINE_FUNCTION float yLocal(int i) const { return m_yl[i]; }

  KOKKOS_INLINE_FUNCTION float& xerrLocal(int i) { return m_xerr[i]; }
  KOKKOS_INLINE_FUNCTION float xerrLocal(int i) const { return m_xerr[i]; }
  KOKKOS_INLINE_FUNCTION float& yerrLocal(int i) { return m_yerr[i]; }
  KOKKOS_INLINE_FUNCTION float yerrLocal(int i) const { return m_yerr[i]; }

  KOKKOS_INLINE_FUNCTION float& xGlobal(int i) { return m_xg[i]; }
  KOKKOS_INLINE_FUNCTION float xGlobal(int i) const { return m_xg[i]; }
  KOKKOS_INLINE_FUNCTION float& yGlobal(int i) { return m_yg[i]; }
  KOKKOS_INLINE_FUNCTION float yGlobal(int i) const { return m_yg[i]; }
  KOKKOS_INLINE_FUNCTION float& zGlobal(int i) { return m_zg[i]; }
  KOKKOS_INLINE_FUNCTION float zGlobal(int i) const { return m_zg[i]; }
  KOKKOS_INLINE_FUNCTION float& rGlobal(int i) { return m_rg[i]; }
  KOKKOS_INLINE_FUNCTION float rGlobal(int i) const { return m_rg[i]; }

  KOKKOS_INLINE_FUNCTION int16_t& iphi(int i) { return m_iphi[i]; }
  KOKKOS_INLINE_FUNCTION int16_t iphi(int i) const { return m_iphi[i]; }

  KOKKOS_INLINE_FUNCTION int32_t& charge(int i) { return m_charge[i]; }
  KOKKOS_INLINE_FUNCTION int32_t charge(int i) const { return m_charge[i]; }
  KOKKOS_INLINE_FUNCTION int16_t& clusterSizeX(int i) { return m_xsize[i]; }
  KOKKOS_INLINE_FUNCTION int16_t clusterSizeX(int i) const { return m_xsize[i]; }
  KOKKOS_INLINE_FUNCTION int16_t& clusterSizeY(int i) { return m_ysize[i]; }
  KOKKOS_INLINE_FUNCTION int16_t clusterSizeY(int i) const { return m_ysize[i]; }
  KOKKOS_INLINE_FUNCTION uint16_t& detectorIndex(int i) { return m_detInd[i]; }
  KOKKOS_INLINE_FUNCTION uint16_t detectorIndex(int i) const { return m_detInd[i]; }

  KOKKOS_INLINE_FUNCTION pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  KOKKOS_INLINE_FUNCTION uint32_t hitsModuleStart(int i) const { return m_hitsModuleStart[i]; }

  KOKKOS_INLINE_FUNCTION uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  KOKKOS_INLINE_FUNCTION uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  KOKKOS_INLINE_FUNCTION Hist& phiBinner() { return *m_hist; }
  KOKKOS_INLINE_FUNCTION Hist const& phiBinner() const { return *m_hist; }

  KOKKOS_INLINE_FUNCTION AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  KOKKOS_INLINE_FUNCTION AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

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
