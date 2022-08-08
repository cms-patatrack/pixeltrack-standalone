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

using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous;

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
