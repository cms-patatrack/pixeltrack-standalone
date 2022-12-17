#ifndef SYCLDataFormats_TrackingRecHit_interface_TrackingRecHit2DSYCL_h
#define SYCLDataFormats_TrackingRecHit_interface_TrackingRecHit2DSYCL_h

#include "SYCLDataFormats/TrackingRecHit2DSOAView.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

class TrackingRecHit2DSYCL {
public:
  using Hist = TrackingRecHit2DSOAView::Hist;

  TrackingRecHit2DSYCL() = default;

  explicit TrackingRecHit2DSYCL(uint32_t nHits,
                                pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                uint32_t const* hitsModuleStart,
                                sycl::queue stream);

  ~TrackingRecHit2DSYCL() = default;

  TrackingRecHit2DSYCL(const TrackingRecHit2DSYCL&) = delete;
  TrackingRecHit2DSYCL& operator=(const TrackingRecHit2DSYCL&) = delete;
  TrackingRecHit2DSYCL(TrackingRecHit2DSYCL&&) = default;
  TrackingRecHit2DSYCL& operator=(TrackingRecHit2DSYCL&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_hist; }
  auto iphi() { return m_iphi; }

  // only the local coord and detector index
  cms::sycltools::host::unique_ptr<float[]> localCoordToHostAsync(sycl::queue stream) const;
  cms::sycltools::host::unique_ptr<uint16_t[]> detIndexToHostAsync(sycl::queue stream) const;
  cms::sycltools::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(sycl::queue stream) const;

  // for validation
  cms::sycltools::host::unique_ptr<float[]> globalCoordToHostAsync(sycl::queue stream) const;
  cms::sycltools::host::unique_ptr<int32_t[]> chargeToHostAsync(sycl::queue stream) const;
  cms::sycltools::host::unique_ptr<int16_t[]> sizeToHostAsync(sycl::queue stream) const;

private:
  static constexpr uint32_t n16 = 4;
  static constexpr uint32_t n32 = 9;
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  cms::sycltools::device::unique_ptr<uint16_t[]> m_store16;  //!
  cms::sycltools::device::unique_ptr<float[]> m_store32;     //!

  cms::sycltools::device::unique_ptr<TrackingRecHit2DSOAView::Hist> m_HistStore;                        //!
  cms::sycltools::device::unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  cms::sycltools::device::unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  Hist* m_hist;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};

inline TrackingRecHit2DSYCL::TrackingRecHit2DSYCL(uint32_t nHits,
                                                  pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                  uint32_t const* hitsModuleStart,
                                                  sycl::queue stream)
    : m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  auto view = cms::sycltools::make_host_unique<TrackingRecHit2DSOAView>(stream);

  view->m_nHits = nHits;
  m_view = cms::sycltools::make_device_unique<TrackingRecHit2DSOAView>(stream);
  m_AverageGeometryStore = cms::sycltools::make_device_unique<TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    stream.memcpy(m_view.get(), view.get(), sizeof(TrackingRecHit2DSOAView));
#ifdef CPU_DEBUG
   stream.wait();
#endif
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated
  m_store16 = cms::sycltools::make_device_unique<uint16_t[]>(nHits * n16, stream);
  m_store32 = cms::sycltools::make_device_unique<float[]>(nHits * n32 + 11, stream);
  m_HistStore = cms::sycltools::make_device_unique<TrackingRecHit2DSOAView::Hist>(stream);

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
  stream.memcpy(m_view.get(), view.get(), sizeof(TrackingRecHit2DSOAView)); 
#ifdef CPU_DEBUG
  stream.wait();
#endif
}

#endif  // SYCLDataFormats_TrackingRecHit_interface_TrackingRecHit2DSYCL_h
