#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "AlpakaDataFormats/TrackingRecHit2DSOAView.h"
#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackingRecHit2DAlpaka {
  public:
    using Hist = TrackingRecHit2DSOAView::Hist;

    TrackingRecHit2DAlpaka() = default;

    explicit TrackingRecHit2DAlpaka(uint32_t nHits,
                                    const pixelCPEforGPU::ParamsOnGPU* cpeParams,
                                    const uint32_t* hitsModuleStart)
        : m_nHits(nHits),
          // NON-OWNING DEVICE POINTERS:
          m_hitsModuleStart(hitsModuleStart),
          // OWNING DEVICE POINTERS:
          m_xl{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_yl{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_xerr{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_yerr{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_xg{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_yg{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_zg{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_rg{cms::alpakatools::allocDeviceBuf<float>(nHits)},
          m_iphi{cms::alpakatools::allocDeviceBuf<int16_t>(nHits)},
          m_charge{cms::alpakatools::allocDeviceBuf<int32_t>(nHits)},
          m_xsize{cms::alpakatools::allocDeviceBuf<int16_t>(nHits)},
          m_ysize{cms::alpakatools::allocDeviceBuf<int16_t>(nHits)},
          m_detInd{cms::alpakatools::allocDeviceBuf<uint16_t>(nHits)},
          m_averageGeometry{cms::alpakatools::allocDeviceBuf<TrackingRecHit2DSOAView::AverageGeometry>(1u)},
          m_hitsLayerStart{cms::alpakatools::allocDeviceBuf<uint32_t>(nHits)},
          m_hist{cms::alpakatools::allocDeviceBuf<Hist>(1u)},
          // SOA view:
          m_view{cms::alpakatools::allocDeviceBuf<TrackingRecHit2DSOAView>(1u)} {
      // the hits are actually accessed in order only in building
      // if ordering is relevant they may have to be stored phi-ordered by layer or so
      // this will break 1to1 correspondence with cluster and module locality
      // so unless proven VERY inefficient we keep it ordered as generated

      // Copy data to the SOA view:
      TrackingRecHit2DSOAView view;
      // By value.
      view.m_nHits = nHits;
      // Raw pointer to data already owned in the event by SiPixelClusterAlpaka object:
      view.m_hitsModuleStart = hitsModuleStart;
      // Raw pointer to data already owned in the eventSetup by PixelCPEFast object:
      view.m_cpeParams = cpeParams;

      // Raw pointers to data owned here in TrackingRecHit2DAlpaka object:
#define SET(name) view.name = alpaka::getPtrNative(name)
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
      SET(m_averageGeometry);
      SET(m_hitsLayerStart);
      SET(m_hist);
#undef SET

      // SoA view on device:
      Queue queue(device);
      auto view_h{cms::alpakatools::createHostView<TrackingRecHit2DSOAView>(&view, 1u)};
      alpaka::memcpy(queue, m_view, view_h, 1u);
      alpaka::wait(queue);
    }

    ~TrackingRecHit2DAlpaka() = default;

    TrackingRecHit2DAlpaka(const TrackingRecHit2DAlpaka&) = delete;
    TrackingRecHit2DAlpaka& operator=(const TrackingRecHit2DAlpaka&) = delete;
    TrackingRecHit2DAlpaka(TrackingRecHit2DAlpaka&&) = default;
    TrackingRecHit2DAlpaka& operator=(TrackingRecHit2DAlpaka&&) = default;

    TrackingRecHit2DSOAView* view() { return alpaka::getPtrNative(m_view); }
    TrackingRecHit2DSOAView const* view() const { return alpaka::getPtrNative(m_view); }

    auto nHits() const { return m_nHits; }
    auto hitsModuleStart() const { return m_hitsModuleStart; }

    auto hitsLayerStart() { return alpaka::getPtrNative(m_hitsLayerStart); }
    auto const* c_hitsLayerStart() const { return alpaka::getPtrNative(m_hitsLayerStart); }
    auto phiBinner() { return alpaka::getPtrNative(m_hist); }
    auto iphi() { return alpaka::getPtrNative(m_iphi); }
    auto const* c_iphi() const { return alpaka::getPtrNative(m_iphi); }

    auto xlToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_xl, nHits());
      return ret;
    }
    auto ylToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_yl, nHits());
      return ret;
    }
    auto xerrToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_xerr, nHits());
      return ret;
    }
    auto yerrToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_yerr, nHits());
      return ret;
    }
    auto xgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_xg, nHits());
      return ret;
    }
    auto ygToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_yg, nHits());
      return ret;
    }
    auto zgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_zg, nHits());
      return ret;
    }
    auto rgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<float>(nHits());
      alpaka::memcpy(queue, ret, m_rg, nHits());
      return ret;
    }
    auto chargeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<int32_t>(nHits());
      alpaka::memcpy(queue, ret, m_charge, nHits());
      return ret;
    }
    auto xsizeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<int16_t>(nHits());
      alpaka::memcpy(queue, ret, m_xsize, nHits());
      return ret;
    }
    auto ysizeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::allocHostBuf<int16_t>(nHits());
      alpaka::memcpy(queue, ret, m_ysize, nHits());
      return ret;
    }
#ifdef TODO
    // only the local coord and detector index
    cms::cuda::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
    cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
#endif
    auto const* xl() const { return alpaka::getPtrNative(m_xl); }
    auto const* yl() const { return alpaka::getPtrNative(m_yl); }
    auto const* xerr() const { return alpaka::getPtrNative(m_xerr); }
    auto const* yerr() const { return alpaka::getPtrNative(m_yerr); }
    auto const* xg() const { return alpaka::getPtrNative(m_xg); }
    auto const* yg() const { return alpaka::getPtrNative(m_yg); }
    auto const* zg() const { return alpaka::getPtrNative(m_zg); }
    auto const* rg() const { return alpaka::getPtrNative(m_rg); }
    auto const* charge() const { return alpaka::getPtrNative(m_charge); }
    auto const* xsize() const { return alpaka::getPtrNative(m_xsize); }
    auto const* ysize() const { return alpaka::getPtrNative(m_ysize); }

  private:
    uint32_t m_nHits;

    // NON-OWNING DEVICE POINTERS
    // m_hitsModuleStart data is already owned by SiPixelClusterAlpaka, let's not abuse of shared_ptr!!
    uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

    // OWNING DEVICE POINTERS
    // local coord
    AlpakaDeviceBuf<float> m_xl;
    AlpakaDeviceBuf<float> m_yl;
    AlpakaDeviceBuf<float> m_xerr;
    AlpakaDeviceBuf<float> m_yerr;

    // global coord
    AlpakaDeviceBuf<float> m_xg;
    AlpakaDeviceBuf<float> m_yg;
    AlpakaDeviceBuf<float> m_zg;
    AlpakaDeviceBuf<float> m_rg;
    AlpakaDeviceBuf<int16_t> m_iphi;

    // cluster properties
    AlpakaDeviceBuf<int32_t> m_charge;
    AlpakaDeviceBuf<int16_t> m_xsize;
    AlpakaDeviceBuf<int16_t> m_ysize;
    AlpakaDeviceBuf<uint16_t> m_detInd;

    AlpakaDeviceBuf<TrackingRecHit2DSOAView::AverageGeometry> m_averageGeometry;

    // needed as kernel params...
    AlpakaDeviceBuf<uint32_t> m_hitsLayerStart;
    AlpakaDeviceBuf<Hist> m_hist;

    // This is a SoA view which itself gathers non-owning pointers to the data owned above (in TrackingRecHit2DAlpaka instance).
    // This is used to access and modify data on GPU in a SoA format (TrackingRecHit2DSOAView),
    // while the data itself is owned here in the TrackingRecHit2DAlpaka instance.
    AlpakaDeviceBuf<TrackingRecHit2DSOAView> m_view;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
