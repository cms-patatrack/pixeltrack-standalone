#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2D_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2D_h

#include "CUDADataFormats/TrackingRecHit2DSOAView.h"

class TrackingRecHit2DHeterogeneous {
public:
  using Hist = TrackingRecHit2DSOAView::Hist;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                         pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                         uint32_t const* hitsModuleStart);

  ~TrackingRecHit2DHeterogeneous() = default;

  TrackingRecHit2DHeterogeneous(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous& operator=(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous(TrackingRecHit2DHeterogeneous&&) = default;
  TrackingRecHit2DHeterogeneous& operator=(TrackingRecHit2DHeterogeneous&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_hist; }
  auto iphi() { return m_iphi; }

  const float* localCoord() const { return m_store32.get(); }
  const float* globalCoord() const { return m_store32.get() + 4 * nHits(); }
  const int32_t* charge() const { return reinterpret_cast<int32_t*>(m_store32.get() + 8 * nHits()); }
  const int16_t* size() const { return reinterpret_cast<int16_t*>(m_store16.get() + 2 * nHits()); }

private:
  static constexpr uint32_t n16 = 4;
  static constexpr uint32_t n32 = 9;
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  std::unique_ptr<uint16_t[]> m_store16;  //!
  std::unique_ptr<float[]> m_store32;     //!

  std::unique_ptr<TrackingRecHit2DSOAView::Hist> m_HistStore;                        //!
  std::unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  std::unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  Hist* m_hist {nullptr};
  uint32_t* m_hitsLayerStart {nullptr};
  int16_t* m_iphi {nullptr};
};

TrackingRecHit2DHeterogeneous::TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                                             pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                             uint32_t const* hitsModuleStart)
    : m_view{std::make_unique<TrackingRecHit2DSOAView>()}, m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  m_view->m_nHits = nHits;
  m_AverageGeometryStore = std::make_unique<TrackingRecHit2DSOAView::AverageGeometry>();
  m_view->m_averageGeometry = m_AverageGeometryStore.get();
  m_view->m_cpeParams = cpeParams;
  m_view->m_hitsModuleStart = hitsModuleStart;

  if (nHits > 0) {
    // the single arrays are not 128 bit alligned...
    // the hits are actually accessed in order only in building
    // if ordering is relevant they may have to be stored phi-ordered by layer or so
    // this will break 1to1 correspondence with cluster and module locality
    // so unless proven VERY inefficient we keep it ordered as generated
    m_store16 = std::make_unique<uint16_t[]>(nHits * n16);
    m_store32 = std::make_unique<float[]>(nHits * n32 + 11);
    m_HistStore = std::make_unique<TrackingRecHit2DSOAView::Hist>();

    auto get16 = [&](int i) { return m_store16.get() + i * nHits; };
    auto get32 = [&](int i) { return m_store32.get() + i * nHits; };

    // copy all the pointers
    m_hist = m_view->m_hist = m_HistStore.get();

    m_view->m_xl = get32(0);
    m_view->m_yl = get32(1);
    m_view->m_xerr = get32(2);
    m_view->m_yerr = get32(3);

    m_view->m_xg = get32(4);
    m_view->m_yg = get32(5);
    m_view->m_zg = get32(6);
    m_view->m_rg = get32(7);

    m_iphi = m_view->m_iphi = reinterpret_cast<int16_t*>(get16(0));

    m_view->m_charge = reinterpret_cast<int32_t*>(get32(8));
    m_view->m_xsize = reinterpret_cast<int16_t*>(get16(2));
    m_view->m_ysize = reinterpret_cast<int16_t*>(get16(3));
    m_view->m_detInd = get16(1);

    m_hitsLayerStart = m_view->m_hitsLayerStart = reinterpret_cast<uint32_t*>(get32(n32));
  }
}

#ifdef CUDAUVM_DISABLE_MANAGED_RECHIT
using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous;
#else
using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous;
#endif
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous;

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
