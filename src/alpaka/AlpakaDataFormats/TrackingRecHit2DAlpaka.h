#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include <memory>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/TrackingRecHit2DSOAView.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackingRecHit2DAlpaka {
  public:
    using Hist = TrackingRecHit2DSOAView::Hist;

    TrackingRecHit2DAlpaka() = default;

    explicit TrackingRecHit2DAlpaka(uint32_t nHits,
                                    const pixelCPEforGPU::ParamsOnGPU* cpeParams,
                                    const uint32_t* hitsModuleStart,
                                    Queue& queue)
        : m_nHits(nHits),
          // NON-OWNING DEVICE POINTERS:
          m_hitsModuleStart(hitsModuleStart),
          // OWNING DEVICE POINTERS:
          m_xl{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_yl{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_xerr{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_yerr{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_xg{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_yg{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_zg{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_rg{::cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_iphi{::cms::alpakatools::make_device_buffer<int16_t[]>(queue, nHits)},
          m_charge{::cms::alpakatools::make_device_buffer<int32_t[]>(queue, nHits)},
          m_xsize{::cms::alpakatools::make_device_buffer<int16_t[]>(queue, nHits)},
          m_ysize{::cms::alpakatools::make_device_buffer<int16_t[]>(queue, nHits)},
          m_detInd{::cms::alpakatools::make_device_buffer<uint16_t[]>(queue, nHits)},
          m_averageGeometry{::cms::alpakatools::make_device_buffer<TrackingRecHit2DSOAView::AverageGeometry>(queue)},
          m_hitsLayerStart{::cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nHits)},
          m_hist{::cms::alpakatools::make_device_buffer<Hist>(queue)},
          // SoA view:
          m_view{::cms::alpakatools::make_device_buffer<TrackingRecHit2DSOAView>(queue)},
          m_view_h{::cms::alpakatools::make_host_buffer<TrackingRecHit2DSOAView>()} {
      // the hits are actually accessed in order only in building
      // if ordering is relevant they may have to be stored phi-ordered by layer or so
      // this will break 1to1 correspondence with cluster and module locality
      // so unless proven VERY inefficient we keep it ordered as generated

      // Copy data to the SoA view:
      TrackingRecHit2DSOAView& view = *alpaka::getPtrNative(m_view_h);
      // By value:
      view.m_nHits = nHits;
      // Raw pointer to data already owned in the event by SiPixelClusterAlpaka object:
      view.m_hitsModuleStart = hitsModuleStart;
      // Raw pointer to data already owned in the eventSetup by PixelCPEFast object:
      view.m_cpeParams = cpeParams;
      // Raw pointers to data owned here in TrackingRecHit2DAlpaka object:
      view.m_xl = alpaka::getPtrNative(m_xl);
      view.m_yl = alpaka::getPtrNative(m_yl);
      view.m_xerr = alpaka::getPtrNative(m_xerr);
      view.m_yerr = alpaka::getPtrNative(m_yerr);
      view.m_xg = alpaka::getPtrNative(m_xg);
      view.m_yg = alpaka::getPtrNative(m_yg);
      view.m_zg = alpaka::getPtrNative(m_zg);
      view.m_rg = alpaka::getPtrNative(m_rg);
      view.m_iphi = alpaka::getPtrNative(m_iphi);
      view.m_charge = alpaka::getPtrNative(m_charge);
      view.m_xsize = alpaka::getPtrNative(m_xsize);
      view.m_ysize = alpaka::getPtrNative(m_ysize);
      view.m_detInd = alpaka::getPtrNative(m_detInd);
      view.m_averageGeometry = alpaka::getPtrNative(m_averageGeometry);
      view.m_hitsLayerStart = alpaka::getPtrNative(m_hitsLayerStart);
      view.m_hist = alpaka::getPtrNative(m_hist);
      // Copy the SoA view to the device
      alpaka::memcpy(queue, m_view, m_view_h);
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
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_xl);
      return ret;
    }
    auto ylToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_yl);
      return ret;
    }
    auto xerrToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_xerr);
      return ret;
    }
    auto yerrToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_yerr);
      return ret;
    }
    auto xgToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_xg);
      return ret;
    }
    auto ygToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_yg);
      return ret;
    }
    auto zgToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_zg);
      return ret;
    }
    auto rgToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_rg);
      return ret;
    }
    auto chargeToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<int32_t[]>(nHits());
      alpaka::memcpy(queue, ret, m_charge);
      return ret;
    }
    auto xsizeToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<int16_t[]>(nHits());
      alpaka::memcpy(queue, ret, m_xsize);
      return ret;
    }
    auto ysizeToHostAsync(Queue& queue) const {
      auto ret = ::cms::alpakatools::make_host_buffer<int16_t[]>(nHits());
      alpaka::memcpy(queue, ret, m_ysize);
      return ret;
    }
#ifdef TODO
    // only the local coord and detector index
    ::cms::alpakatools::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
    ::cms::alpakatools::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
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
    ::cms::alpakatools::device_buffer<Device, float[]> m_xl;
    ::cms::alpakatools::device_buffer<Device, float[]> m_yl;
    ::cms::alpakatools::device_buffer<Device, float[]> m_xerr;
    ::cms::alpakatools::device_buffer<Device, float[]> m_yerr;

    // global coord
    ::cms::alpakatools::device_buffer<Device, float[]> m_xg;
    ::cms::alpakatools::device_buffer<Device, float[]> m_yg;
    ::cms::alpakatools::device_buffer<Device, float[]> m_zg;
    ::cms::alpakatools::device_buffer<Device, float[]> m_rg;
    ::cms::alpakatools::device_buffer<Device, int16_t[]> m_iphi;

    // cluster properties
    ::cms::alpakatools::device_buffer<Device, int32_t[]> m_charge;
    ::cms::alpakatools::device_buffer<Device, int16_t[]> m_xsize;
    ::cms::alpakatools::device_buffer<Device, int16_t[]> m_ysize;
    ::cms::alpakatools::device_buffer<Device, uint16_t[]> m_detInd;

    ::cms::alpakatools::device_buffer<Device, TrackingRecHit2DSOAView::AverageGeometry> m_averageGeometry;

    // needed as kernel params...
    ::cms::alpakatools::device_buffer<Device, uint32_t[]> m_hitsLayerStart;
    ::cms::alpakatools::device_buffer<Device, Hist> m_hist;

    // This is a SoA view which itself gathers non-owning pointers to the data owned above (in TrackingRecHit2DAlpaka instance).
    // This is used to access and modify data on GPU in a SoA format (TrackingRecHit2DSOAView),
    // while the data itself is owned here in the TrackingRecHit2DAlpaka instance.
    ::cms::alpakatools::device_buffer<Device, TrackingRecHit2DSOAView> m_view;
    // Keep a host copy of the device view alive during the asynchronous copy
    ::cms::alpakatools::host_buffer<TrackingRecHit2DSOAView> m_view_h;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
