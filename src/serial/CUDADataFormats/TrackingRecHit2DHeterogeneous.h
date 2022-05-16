#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/HeterogeneousSoA.h"
#include <vector>
#include <iostream>

template <typename Traits>
class TrackingRecHit2DHeterogeneous {
public:
  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using Hist = TrackingRecHit2DSOAView::Hist;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                         pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                         uint32_t const* hitsModuleStart,
                                         cudaStream_t stream);
  TrackingRecHit2DHeterogeneous(std::vector<float>& x_coord,
                                std::vector<float>& y_coord, 
                                std::vector<float>& z_coord, 
                                std::vector<float>& r_coord,
                                std::vector<uint32_t>& layerStart,
                                std::vector<int>& phi,
                                cudaStream_t stream);

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

  //TrackingRecHit2DSOAView getView() const { return view_; }

private:
  static constexpr uint32_t n16 = 0;
  static constexpr uint32_t n32 = 9;
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  unique_ptr<uint16_t[]> m_store16;  //!
  unique_ptr<float[]> m_store32;     //!

  unique_ptr<TrackingRecHit2DSOAView::Hist> m_HistStore;                        //!
  unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  Hist* m_hist;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;

  int event_number;

  //TrackingRecHit2DSOAView view_;
};

template <typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                                                     pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                                     uint32_t const* hitsModuleStart,
                                                                     cudaStream_t stream)
    : m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);

  view->m_nHits = nHits;
  m_view = Traits::template make_device_unique<TrackingRecHit2DSOAView>(stream);
  m_AverageGeometryStore = Traits::template make_device_unique<TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
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
  m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
}

// My constructor
template <typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(std::vector<float>& x_coord,
                                                                     std::vector<float>& y_coord, 
                                                                     std::vector<float>& z_coord, 
                                                                     std::vector<float>& r_coord,
                                                                     std::vector<uint32_t>& layerStart,
                                                                     std::vector<int>& phi,
                                                                     cudaStream_t stream)
    : m_nHits(x_coord.size()) {
  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);
  
  view->m_nHits = x_coord.size();
  m_HistStore = Traits::template make_device_unique<TrackingRecHit2DSOAView::Hist>(stream);
  m_hist = view->m_hist = m_HistStore.get();
  //std::cout << "mhist size" << m_HistStore->capacity() << '\n';
  //m_hist = view->m_hist = (Hist*)malloc(m_HistStore->capacity()*sizeof(uint32_t));
 
  view->m_xg = (float*)malloc(x_coord.size()*sizeof(float));
  for(int j = 0; j < static_cast<int>(x_coord.size()); ++j) {
    view->m_xg[j] = x_coord[j];
  }
  view->m_yg = (float*)malloc(x_coord.size()*sizeof(float));
  for(int j = 0; j < static_cast<int>(x_coord.size()); ++j) {
    view->m_yg[j] = y_coord[j];
  }
  view->m_zg = (float*)malloc(x_coord.size()*sizeof(float));
  for(int j = 0; j < static_cast<int>(x_coord.size()); ++j) {
    view->m_zg[j] = z_coord[j];
  }
  view->m_rg = (float*)malloc(x_coord.size()*sizeof(float));
  for(int j = 0; j < static_cast<int>(x_coord.size()); ++j) {
    view->m_rg[j] = r_coord[j];
  }

  //view->m_hitsLayerStart = layerStart.data();
  view->m_hitsLayerStart = (uint32_t*)malloc(layerStart.size()*sizeof(uint32_t));
  for(int j = 0; j < static_cast<int>(layerStart.size()); ++j) {
    view->m_hitsLayerStart[j] = layerStart[j];
  }

  m_hitsLayerStart = layerStart.data();

  view->m_iphi = (int16_t*)malloc(x_coord.size()*sizeof(int16_t));
  for(int j = 0; j < static_cast<int>(x_coord.size()); ++j) {
    view->m_iphi[j] = phi[j];
  }

  std::cout << "costruttore finito" << '\n';
  m_view.reset(view.release()); 
}

using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::CPUTraits>;

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
