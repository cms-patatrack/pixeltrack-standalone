#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include <memory>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaDataFormats/TrackingRecHit2DSoAView.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackingRecHit2DAlpaka {
  public:
    using Hist = TrackingRecHit2DSoAView::Hist;

    TrackingRecHit2DAlpaka() = default;

    explicit TrackingRecHit2DAlpaka(uint32_t nHits,
                                    const pixelCPEforGPU::ParamsOnGPU* cpeParams,
                                    const uint32_t* hitsModuleStart,
                                    Queue& queue)
        : m_nHits(nHits),
          // NON-OWNING DEVICE POINTERS:
          m_hitsModuleStart(hitsModuleStart),
          // OWNING DEVICE POINTERS:
          m_xl{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_yl{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_xerr{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_yerr{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_xg{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_yg{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_zg{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_rg{cms::alpakatools::make_device_buffer<float[]>(queue, nHits)},
          m_iphi{cms::alpakatools::make_device_buffer<int16_t[]>(queue, nHits)},
          m_charge{cms::alpakatools::make_device_buffer<int32_t[]>(queue, nHits)},
          m_xsize{cms::alpakatools::make_device_buffer<int16_t[]>(queue, nHits)},
          m_ysize{cms::alpakatools::make_device_buffer<int16_t[]>(queue, nHits)},
          m_detInd{cms::alpakatools::make_device_buffer<uint16_t[]>(queue, nHits)},
          m_averageGeometry{cms::alpakatools::make_device_buffer<TrackingRecHit2DSoAView::AverageGeometry>(queue)},
          m_hitsLayerStart{cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nHits)},
          m_hist{cms::alpakatools::make_device_buffer<Hist>(queue)},
          // SoA view:
          m_view{cms::alpakatools::make_device_buffer<TrackingRecHit2DSoAView>(queue)},
          m_view_h{cms::alpakatools::make_host_buffer<TrackingRecHit2DSoAView>()} {
      // the hits are actually accessed in order only in building
      // if ordering is relevant they may have to be stored phi-ordered by layer or so
      // this will break 1to1 correspondence with cluster and module locality
      // so unless proven VERY inefficient we keep it ordered as generated

      // Copy data to the SoA view:
      // By value:
      m_view_h->m_nHits = nHits;
      // Raw pointer to data already owned in the event by SiPixelClusterAlpaka object:
      m_view_h->m_hitsModuleStart = hitsModuleStart;
      // Raw pointer to data already owned in the eventSetup by PixelCPEFast object:
      m_view_h->m_cpeParams = cpeParams;
      // Raw pointers to data owned here in TrackingRecHit2DAlpaka object:
      m_view_h->m_xl = m_xl.data();
      m_view_h->m_yl = m_yl.data();
      m_view_h->m_xerr = m_xerr.data();
      m_view_h->m_yerr = m_yerr.data();
      m_view_h->m_xg = m_xg.data();
      m_view_h->m_yg = m_yg.data();
      m_view_h->m_zg = m_zg.data();
      m_view_h->m_rg = m_rg.data();
      m_view_h->m_iphi = m_iphi.data();
      m_view_h->m_charge = m_charge.data();
      m_view_h->m_xsize = m_xsize.data();
      m_view_h->m_ysize = m_ysize.data();
      m_view_h->m_detInd = m_detInd.data();
      m_view_h->m_averageGeometry = m_averageGeometry.data();
      m_view_h->m_hitsLayerStart = m_hitsLayerStart.data();
      m_view_h->m_hist = m_hist.data();
      // Copy the SoA view to the device
      alpaka::memcpy(queue, m_view, m_view_h);
    }

    ~TrackingRecHit2DAlpaka() = default;

    TrackingRecHit2DAlpaka(const TrackingRecHit2DAlpaka&) = delete;
    TrackingRecHit2DAlpaka& operator=(const TrackingRecHit2DAlpaka&) = delete;
    TrackingRecHit2DAlpaka(TrackingRecHit2DAlpaka&&) = default;
    TrackingRecHit2DAlpaka& operator=(TrackingRecHit2DAlpaka&&) = default;

    TrackingRecHit2DSoAView* view() { return m_view.data(); }
    TrackingRecHit2DSoAView const* view() const { return m_view.data(); }

    auto nHits() const { return m_nHits; }
    auto hitsModuleStart() const { return m_hitsModuleStart; }

    auto hitsLayerStart() { return m_hitsLayerStart.data(); }
    auto const* c_hitsLayerStart() const { return m_hitsLayerStart.data(); }
    auto phiBinner() { return m_hist.data(); }
    auto iphi() { return m_iphi.data(); }
    auto const* c_iphi() const { return m_iphi.data(); }

    auto xlToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_xl);
      return ret;
    }
    auto ylToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_yl);
      return ret;
    }
    auto xerrToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_xerr);
      return ret;
    }
    auto yerrToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_yerr);
      return ret;
    }
    auto xgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_xg);
      return ret;
    }
    auto ygToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_yg);
      return ret;
    }
    auto zgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_zg);
      return ret;
    }
    auto rgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<float[]>(nHits());
      alpaka::memcpy(queue, ret, m_rg);
      return ret;
    }
    auto chargeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<int32_t[]>(nHits());
      alpaka::memcpy(queue, ret, m_charge);
      return ret;
    }
    auto xsizeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<int16_t[]>(nHits());
      alpaka::memcpy(queue, ret, m_xsize);
      return ret;
    }
    auto ysizeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_buffer<int16_t[]>(nHits());
      alpaka::memcpy(queue, ret, m_ysize);
      return ret;
    }
#ifdef TODO
    // only the local coord and detector index
    cms::alpakatools::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
    cms::alpakatools::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
#endif
    auto const* xl() const { return m_xl.data(); }
    auto const* yl() const { return m_yl.data(); }
    auto const* xerr() const { return m_xerr.data(); }
    auto const* yerr() const { return m_yerr.data(); }
    auto const* xg() const { return m_xg.data(); }
    auto const* yg() const { return m_yg.data(); }
    auto const* zg() const { return m_zg.data(); }
    auto const* rg() const { return m_rg.data(); }
    auto const* charge() const { return m_charge.data(); }
    auto const* xsize() const { return m_xsize.data(); }
    auto const* ysize() const { return m_ysize.data(); }

  private:
    uint32_t m_nHits;

    // NON-OWNING DEVICE POINTERS
    // m_hitsModuleStart data is already owned by SiPixelClusterAlpaka, let's not abuse of shared_ptr!!
    uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

    // OWNING DEVICE POINTERS
    // local coord
    cms::alpakatools::device_buffer<Device, float[]> m_xl;
    cms::alpakatools::device_buffer<Device, float[]> m_yl;
    cms::alpakatools::device_buffer<Device, float[]> m_xerr;
    cms::alpakatools::device_buffer<Device, float[]> m_yerr;

    // global coord
    cms::alpakatools::device_buffer<Device, float[]> m_xg;
    cms::alpakatools::device_buffer<Device, float[]> m_yg;
    cms::alpakatools::device_buffer<Device, float[]> m_zg;
    cms::alpakatools::device_buffer<Device, float[]> m_rg;
    cms::alpakatools::device_buffer<Device, int16_t[]> m_iphi;

    // cluster properties
    cms::alpakatools::device_buffer<Device, int32_t[]> m_charge;
    cms::alpakatools::device_buffer<Device, int16_t[]> m_xsize;
    cms::alpakatools::device_buffer<Device, int16_t[]> m_ysize;
    cms::alpakatools::device_buffer<Device, uint16_t[]> m_detInd;

    cms::alpakatools::device_buffer<Device, TrackingRecHit2DSoAView::AverageGeometry> m_averageGeometry;

    // needed as kernel params...
    cms::alpakatools::device_buffer<Device, uint32_t[]> m_hitsLayerStart;
    cms::alpakatools::device_buffer<Device, Hist> m_hist;

    // This is a SoA view which itself gathers non-owning pointers to the data owned above (in TrackingRecHit2DAlpaka instance).
    // This is used to access and modify data on GPU in a SoA format (TrackingRecHit2DSoAView),
    // while the data itself is owned here in the TrackingRecHit2DAlpaka instance.
    cms::alpakatools::device_buffer<Device, TrackingRecHit2DSoAView> m_view;
    // Keep a host copy of the device view alive during the asynchronous copy
    cms::alpakatools::host_buffer<TrackingRecHit2DSoAView> m_view_h;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
