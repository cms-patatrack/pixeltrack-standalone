#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "KokkosDataFormats/TrackingRecHit2DSOAView.h"
#ifdef TODO
#include "CUDADataFormats/HeterogeneousSoA.h"
#endif

template <typename MemorySpace>
class TrackingRecHit2DKokkos {
public:
#ifdef TODO
  using Hist = TrackingRecHit2DSOAView::Hist;
#endif

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

#ifdef TODO
  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_hist; }
  auto iphi() { return m_iphi; }

  // only the local coord and detector index
  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
#endif

private:
#ifdef TODO
  static constexpr uint32_t n16 = 4;
  static constexpr uint32_t n32 = 9;
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  unique_ptr<uint16_t[]> m_store16;  //!
  unique_ptr<float[]> m_store32;     //!

  unique_ptr<TrackingRecHit2DSOAView::Hist> m_HistStore;                        //!
  unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAView> m_view;  //!
#endif

  uint32_t m_nHits;
  Kokkos::View<uint32_t const*, MemorySpace> m_hitsModuleStart;  // needed for legacy, this is on GPU!
#ifdef TODO

  // needed as kernel params...
  Hist* m_hist;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
#endif
};

template <typename MemorySpace>
template <typename ExecSpace>
TrackingRecHit2DKokkos<MemorySpace>::TrackingRecHit2DKokkos(uint32_t nHits,
                                                            Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, MemorySpace> cpeParams,
                                                            Kokkos::View<uint32_t const*, MemorySpace>  hitsModuleStart,
                                                            ExecSpace const& execSpace)
: m_nHits(nHits), m_hitsModuleStart(std::move(hitsModuleStart)) {
#ifdef TODO
  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);

  view->m_nHits = nHits;
  m_view = Traits::template make_device_unique<TrackingRecHit2DSOAView>(stream);
  m_AverageGeometryStore = Traits::template make_device_unique<TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if
#ifndef __CUDACC__
        constexpr
#endif
        (std::is_same<Traits, cudaCompat::GPUTraits>::value) {
      cms::cuda::copyAsync(m_view, view, stream);
    } else {
      m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
    }
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated
  m_store16 = Traits::template make_device_unique<uint16_t[]>(nHits * n16, stream);
  m_store32 = Traits::template make_device_unique<float[]>(nHits * n32 + 11, stream);
  m_HistStore = Traits::template make_device_unique<TrackingRecHit2DSOAView::Hist>(stream);

  auto get16 = [&](int i) { return m_store16.get() + i * nHits; };
  auto get32 = [&](int i) { return m_store32.get() + i * nHits; };

  // copy all the pointers
  m_hist = view->m_hist = m_HistStore.get();

  view->m_xl = get32(0);
  view->m_yl = get32(1);
  view->m_xerr = get32(2);
  view->m_yerr = get32(3);

  view->m_xg = get32(4);
  view->m_yg = get32(5);
  view->m_zg = get32(6);
  view->m_rg = get32(7);

  m_iphi = view->m_iphi = reinterpret_cast<int16_t*>(get16(0));

  view->m_charge = reinterpret_cast<int32_t*>(get32(8));
  view->m_xsize = reinterpret_cast<int16_t*>(get16(2));
  view->m_ysize = reinterpret_cast<int16_t*>(get16(3));
  view->m_detInd = get16(1);

  m_hitsLayerStart = view->m_hitsLayerStart = reinterpret_cast<uint32_t*>(get32(n32));

  // transfer view
  if
#ifndef __CUDACC__
      constexpr
#endif
      (std::is_same<Traits, cudaCompat::GPUTraits>::value) {
    cms::cuda::copyAsync(m_view, view, stream);
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }
#endif
}

#ifdef TODO
using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous<cudaCompat::GPUTraits>;
using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous<cudaCompat::GPUTraits>;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous<cudaCompat::CPUTraits>;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous<cudaCompat::HostTraits>;
#endif

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
