#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/HeterogeneousSoA.h"

template <typename Traits>
class TrackingRecHit2DHeterogeneous {
public:
  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using PhiBinner = TrackingRecHit2DSOAView::PhiBinner;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                         pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                         uint32_t const* hitsModuleStart,
                                         cms::cuda::Context const& ctx);

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
  auto phiBinner() { return m_phiBinner; }
  auto phiBinnerStorage() { return m_phiBinnerStorage; }
  auto iphi() { return m_iphi; }

  // only the local coord and detector index
  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cms::cuda::Context const& ctx) const;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cms::cuda::Context const& ctx) const;

  // for validation
  cms::cuda::host::unique_ptr<float[]> globalCoordToHostAsync(cms::cuda::Context const& ctx) const;
  cms::cuda::host::unique_ptr<int32_t[]> chargeToHostAsync(cms::cuda::Context const& ctx) const;
  cms::cuda::host::unique_ptr<int16_t[]> sizeToHostAsync(cms::cuda::Context const& ctx) const;

private:
  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;                // number of elements in m_store32
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  unique_ptr<uint16_t[]> m_store16;  //!
  unique_ptr<float[]> m_store32;     //!

  unique_ptr<TrackingRecHit2DSOAView::PhiBinner> m_PhiBinnerStore;              //!
  unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  PhiBinner* m_phiBinner;
  PhiBinner::index_type* m_phiBinnerStorage;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};

#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"

template <typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                                                     pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                                     uint32_t const* hitsModuleStart,
                                                                     cms::cuda::Context const& ctx)
    : m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(ctx);

  view->m_nHits = nHits;
  m_view = Traits::template make_device_unique<TrackingRecHit2DSOAView>(ctx);
  m_AverageGeometryStore = Traits::template make_device_unique<TrackingRecHit2DSOAView::AverageGeometry>(ctx);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
      cms::cuda::copyAsync(m_view, view, ctx.stream());
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
  m_store16 = Traits::template make_device_unique<uint16_t[]>(nHits * n16, ctx);
  m_store32 = Traits::template make_device_unique<float[]>(nHits * n32 + phase1PixelTopology::numberOfLayers + 1, ctx);
  m_PhiBinnerStore = Traits::template make_device_unique<TrackingRecHit2DSOAView::PhiBinner>(ctx);

  static_assert(sizeof(TrackingRecHit2DSOAView::hindex_type) == sizeof(float));
  static_assert(sizeof(TrackingRecHit2DSOAView::hindex_type) == sizeof(TrackingRecHit2DSOAView::PhiBinner::index_type));

  auto get16 = [&](int i) { return m_store16.get() + i * nHits; };
  auto get32 = [&](int i) { return m_store32.get() + i * nHits; };

  // copy all the pointers
  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(9));

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
  if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cms::cuda::copyAsync(m_view, view, ctx.stream());
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }
}

using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits>;
using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits>;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::CPUTraits>;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous<cms::cudacompat::HostTraits>;

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
