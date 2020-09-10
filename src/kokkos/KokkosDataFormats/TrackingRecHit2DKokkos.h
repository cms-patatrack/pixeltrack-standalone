#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "KokkosDataFormats/TrackingRecHit2DSOAView.h"

template <typename MemorySpace>
class TrackingRecHit2DKokkos {
public:
  using Hist = TrackingRecHit2DSOAView::Hist;

  TrackingRecHit2DKokkos() = default;

  template <typename ExecSpace>
  explicit TrackingRecHit2DKokkos(uint32_t nHits,
                                  Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, MemorySpace> cpeParams,
                                  Kokkos::View<uint32_t const*, MemorySpace> hitsModuleStart,
                                  ExecSpace const& execSpace);

  ~TrackingRecHit2DKokkos() = default;

  TrackingRecHit2DKokkos(const TrackingRecHit2DKokkos&) = delete;
  TrackingRecHit2DKokkos& operator=(const TrackingRecHit2DKokkos&) = delete;
  TrackingRecHit2DKokkos(TrackingRecHit2DKokkos&&) = default;
  TrackingRecHit2DKokkos& operator=(TrackingRecHit2DKokkos&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.data(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.data(); }

  auto nHits() const { return m_nHits; }

  Kokkos::View<uint32_t const*, MemorySpace> hitsModuleStart() const { return m_hitsModuleStart; }
  Kokkos::View<uint32_t*, MemorySpace> hitsLayerStart() { return m_hitsLayerStart; }
  Kokkos::View<Hist*, MemorySpace> phiBinner() { return m_hist; }
  Kokkos::View<uint16_t, MemorySpace> iphi() { return m_iphi; }

#ifdef TODO
  // only the local coord and detector index
  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
#endif

private:
  // local coord
  Kokkos::View<float*, MemorySpace> m_xl;
  Kokkos::View<float*, MemorySpace> m_yl;
  Kokkos::View<float*, MemorySpace> m_xerr;
  Kokkos::View<float*, MemorySpace> m_yerr;

  // global coord
  Kokkos::View<float*, MemorySpace> m_xg;
  Kokkos::View<float*, MemorySpace> m_yg;
  Kokkos::View<float*, MemorySpace> m_zg;
  Kokkos::View<float*, MemorySpace> m_rg;
  Kokkos::View<int16_t*, MemorySpace> m_iphi;

  // cluster properties
  Kokkos::View<int32_t*, MemorySpace> m_charge;
  Kokkos::View<int16_t*, MemorySpace> m_xsize;
  Kokkos::View<int16_t*, MemorySpace> m_ysize;
  Kokkos::View<uint16_t*, MemorySpace> m_detInd;

  Kokkos::View<TrackingRecHit2DSOAView::AverageGeometry, MemorySpace> m_AverageGeometryStore;  //!

  Kokkos::View<TrackingRecHit2DSOAView, MemorySpace> m_view;  //!

  uint32_t m_nHits;
  Kokkos::View<uint32_t const*, MemorySpace> m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  Kokkos::View<Hist, MemorySpace> m_hist;
  Kokkos::View<uint32_t*, MemorySpace> m_hitsLayerStart;
};

template <typename MemorySpace>
template <typename ExecSpace>
TrackingRecHit2DKokkos<MemorySpace>::TrackingRecHit2DKokkos(
    uint32_t nHits,
    Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, MemorySpace> cpeParams,
    Kokkos::View<uint32_t const*, MemorySpace> hitsModuleStart,
    ExecSpace const& execSpace)
    : m_nHits(nHits),
      m_xl("m_xl", nHits),
      m_yl("m_yl", nHits),
      m_xerr("m_xerr", nHits),
      m_yerr("m_yerr", nHits),
      m_xg("m_xg", nHits),
      m_yg("m_yg", nHits),
      m_zg("m_zg", nHits),
      m_rg("m_rg", nHits),
      m_iphi("m_iphi", nHits),
      m_charge("m_charge", nHits),
      m_xsize("m_xsize", nHits),
      m_ysize("m_ysize", nHits),
      m_detInd("m_detInd", nHits),
      m_AverageGeometryStore("m_AverageGeometryStore"),
      m_view("m_view"),
      m_hitsModuleStart(std::move(hitsModuleStart)),
      m_hist("m_hist"),
      m_hitsLayerStart("m_hitsLayerStart", nHits) {
  // should I deal with no hits case?

  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

  auto view_h = Kokkos::create_mirror_view(m_view);
#define SET(name) view_h().name = name.data()
  SET(m_xl);
  SET(m_yl);
  SET(m_xerr);
  SET(m_yerr);
  SET(m_xg);
  SET(m_yg);
  SET(m_zg);
  SET(m_rg);
  SET(m_iphi);
  SET(m_charge);
  SET(m_xsize);
  SET(m_ysize);
  SET(m_detInd);
  SET(m_hist);
#undef SET
  view_h().m_nHits = nHits;
  view_h().m_averageGeometry = m_AverageGeometryStore.data();
  view_h().m_cpeParams = cpeParams.data();
  view_h().m_hitsModuleStart = m_hitsModuleStart.data();

  Kokkos::deep_copy(execSpace, m_view, view_h);
}

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
