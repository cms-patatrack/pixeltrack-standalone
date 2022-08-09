#include <memory>

#include "CUDADataFormats/TrackingRecHit2D.h"
#include "CUDADataFormats/TrackingRecHit2DSOAView.h"

TrackingRecHit2D::TrackingRecHit2D(uint32_t nHits,
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