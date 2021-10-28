#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "KokkosCore/ViewHelpers.h"
#include "KokkosCore/deep_copy.h"
#include "KokkosCore/shared_ptr.h"

#include "KokkosDataFormats/TrackingRecHit2DSOAView.h"

template <typename MemorySpace>
class TrackingRecHit2DKokkos {
public:
  using Hist = TrackingRecHit2DSOAView::Hist;
  template <typename T>
  using View = Kokkos::View<T, MemorySpace, RestrictUnmanaged>;

  TrackingRecHit2DKokkos() = default;

  template <typename ExecSpace>
  explicit TrackingRecHit2DKokkos(uint32_t nHits,
                                  Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, MemorySpace> cpeParams,
                                  View<uint32_t const*> hitsModuleStart,
                                  ExecSpace const& execSpace);

  ~TrackingRecHit2DKokkos() = default;

  TrackingRecHit2DKokkos(const TrackingRecHit2DKokkos&) = delete;
  TrackingRecHit2DKokkos& operator=(const TrackingRecHit2DKokkos&) = delete;
  TrackingRecHit2DKokkos(TrackingRecHit2DKokkos&&) = default;
  TrackingRecHit2DKokkos& operator=(TrackingRecHit2DKokkos&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }
  Kokkos::View<TrackingRecHit2DSOAView, MemorySpace, RestrictUnmanaged> mView() { return cms::kokkos::to_view(m_view); }

  auto nHits() const { return m_nHits; }

  View<uint32_t const*> hitsModuleStart() const { return m_hitsModuleStart; }
  View<uint32_t*> hitsLayerStart() { return cms::kokkos::to_view(m_hitsLayerStart); }
  View<Hist> phiBinner() { return cms::kokkos::to_view(m_hist); }
  View<int16_t*> iphi() { return cms::kokkos::to_view(m_iphi); }

  View<uint32_t const*> c_hitsLayerStart() { return cms::kokkos::to_view(m_hitsLayerStart); }
  View<int16_t const*> c_iphi() { return cms::kokkos::to_view(m_iphi); }

#define TO_HOST_ASYNC(name)                                  \
  template <typename ExecSpace>                              \
  auto name##ToHostAsync(ExecSpace const& execSpace) const { \
    auto host = cms::kokkos::make_mirror_shared(m_##name);   \
    cms::kokkos::deep_copy(execSpace, host, m_##name);       \
    return host;                                             \
  }
  TO_HOST_ASYNC(xl);
  TO_HOST_ASYNC(yl);
  TO_HOST_ASYNC(xerr);
  TO_HOST_ASYNC(yerr);
  TO_HOST_ASYNC(xg);
  TO_HOST_ASYNC(yg);
  TO_HOST_ASYNC(zg);
  TO_HOST_ASYNC(rg);
  TO_HOST_ASYNC(charge);
  TO_HOST_ASYNC(xsize);
  TO_HOST_ASYNC(ysize);
#undef TO_HOST_ASYNC

#ifdef TODO
  // only the local coord and detector index
  cms::cuda::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
#endif

private:
  uint32_t m_nHits;

  // local coord
  cms::kokkos::shared_ptr<float[], MemorySpace> m_xl;
  cms::kokkos::shared_ptr<float[], MemorySpace> m_yl;
  cms::kokkos::shared_ptr<float[], MemorySpace> m_xerr;
  cms::kokkos::shared_ptr<float[], MemorySpace> m_yerr;

  // global coord
  cms::kokkos::shared_ptr<float[], MemorySpace> m_xg;
  cms::kokkos::shared_ptr<float[], MemorySpace> m_yg;
  cms::kokkos::shared_ptr<float[], MemorySpace> m_zg;
  cms::kokkos::shared_ptr<float[], MemorySpace> m_rg;
  cms::kokkos::shared_ptr<int16_t[], MemorySpace> m_iphi;

  // cluster properties
  cms::kokkos::shared_ptr<int32_t[], MemorySpace> m_charge;
  cms::kokkos::shared_ptr<int16_t[], MemorySpace> m_xsize;
  cms::kokkos::shared_ptr<int16_t[], MemorySpace> m_ysize;
  cms::kokkos::shared_ptr<uint16_t[], MemorySpace> m_detInd;

  cms::kokkos::shared_ptr<TrackingRecHit2DSOAView::AverageGeometry, MemorySpace> m_AverageGeometryStore;  //!

  cms::kokkos::shared_ptr<TrackingRecHit2DSOAView, MemorySpace> m_view;  //!

  View<uint32_t const*> m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  cms::kokkos::shared_ptr<Hist, MemorySpace> m_hist;
  cms::kokkos::shared_ptr<uint32_t[], MemorySpace> m_hitsLayerStart;
};

template <typename MemorySpace>
template <typename ExecSpace>
TrackingRecHit2DKokkos<MemorySpace>::TrackingRecHit2DKokkos(
    uint32_t nHits,
    Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, MemorySpace> cpeParams,
    View<uint32_t const*> hitsModuleStart,
    ExecSpace const& execSpace)
    : m_nHits(nHits),
      m_xl(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_yl(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_xerr(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_yerr(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_xg(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_yg(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_zg(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_rg(cms::kokkos::make_shared<float[], MemorySpace>(nHits)),
      m_iphi(cms::kokkos::make_shared<int16_t[], MemorySpace>(nHits)),
      m_charge(cms::kokkos::make_shared<int32_t[], MemorySpace>(nHits)),
      m_xsize(cms::kokkos::make_shared<int16_t[], MemorySpace>(nHits)),
      m_ysize(cms::kokkos::make_shared<int16_t[], MemorySpace>(nHits)),
      m_detInd(cms::kokkos::make_shared<uint16_t[], MemorySpace>(nHits)),
      m_AverageGeometryStore(cms::kokkos::make_shared<TrackingRecHit2DSOAView::AverageGeometry, MemorySpace>()),
      m_view(cms::kokkos::make_shared<TrackingRecHit2DSOAView, MemorySpace>()),
      m_hitsModuleStart(std::move(hitsModuleStart)),
      m_hist(cms::kokkos::make_shared<Hist, MemorySpace>()),
      m_hitsLayerStart(cms::kokkos::make_shared<uint32_t[], MemorySpace>(nHits)) {
  // should I deal with no hits case?

  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

  auto view_h = cms::kokkos::make_mirror_shared(m_view);
#define SET(name) view_h->name = name.get()
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
  view_h->m_nHits = nHits;
  view_h->m_averageGeometry = m_AverageGeometryStore.get();
  view_h->m_cpeParams = cpeParams.data();
  view_h->m_hitsModuleStart = m_hitsModuleStart.data();
  view_h->m_hitsLayerStart = m_hitsLayerStart.get();

  cms::kokkos::deep_copy(execSpace, m_view, view_h);
}

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
