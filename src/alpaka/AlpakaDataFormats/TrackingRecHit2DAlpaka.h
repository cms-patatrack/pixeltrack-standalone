#ifndef AlpakaDataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define AlpakaDataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "AlpakaDataFormats/TrackingRecHit2DSOAView.h"
#include "AlpakaCore/device_unique_ptr.h"
#include "AlpakaCore/host_unique_ptr.h"

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
          m_xl{cms::alpakatools::make_device_unique<float>(nHits)},
          m_yl{cms::alpakatools::make_device_unique<float>(nHits)},
          m_xerr{cms::alpakatools::make_device_unique<float>(nHits)},
          m_yerr{cms::alpakatools::make_device_unique<float>(nHits)},
          m_xg{cms::alpakatools::make_device_unique<float>(nHits)},
          m_yg{cms::alpakatools::make_device_unique<float>(nHits)},
          m_zg{cms::alpakatools::make_device_unique<float>(nHits)},
          m_rg{cms::alpakatools::make_device_unique<float>(nHits)},
          m_iphi{cms::alpakatools::make_device_unique<int16_t>(nHits)},
          m_charge{cms::alpakatools::make_device_unique<int32_t>(nHits)},
          m_xsize{cms::alpakatools::make_device_unique<int16_t>(nHits)},
          m_ysize{cms::alpakatools::make_device_unique<int16_t>(nHits)},
          m_detInd{cms::alpakatools::make_device_unique<uint16_t>(nHits)},
          m_averageGeometry{cms::alpakatools::make_device_unique<TrackingRecHit2DSOAView::AverageGeometry>(1u)},
          m_hitsLayerStart{cms::alpakatools::make_device_unique<uint32_t>(nHits)},
          m_hist{cms::alpakatools::make_device_unique<Hist>(1u)},
          // SOA view:
          m_view{cms::alpakatools::make_device_unique<TrackingRecHit2DSOAView>(1u)} {
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
#define SET(name) view.name = name.get()
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
      Queue queue{device};

      // SoA view on device:
      auto view_h{cms::alpakatools::createHostView<TrackingRecHit2DSOAView>(&view, 1u)};
      auto view_m_view{cms::alpakatools::createDeviceView<TrackingRecHit2DSOAView>(m_view.get(), 1u)};
      alpaka::memcpy(queue, view_m_view, view_h, 1u);
      alpaka::wait(queue);
    }

    ~TrackingRecHit2DAlpaka() = default;

    TrackingRecHit2DAlpaka(const TrackingRecHit2DAlpaka&) = delete;
    TrackingRecHit2DAlpaka& operator=(const TrackingRecHit2DAlpaka&) = delete;
    TrackingRecHit2DAlpaka(TrackingRecHit2DAlpaka&&) = default;
    TrackingRecHit2DAlpaka& operator=(TrackingRecHit2DAlpaka&&) = default;

    TrackingRecHit2DSOAView* view() { return m_view.get(); }
    TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

    auto nHits() const { return m_nHits; }
    auto hitsModuleStart() const { return m_hitsModuleStart; }

    auto hitsLayerStart() { return m_hitsLayerStart.get(); }
    auto const* c_hitsLayerStart() const { return m_hitsLayerStart.get(); }
    auto phiBinner() { return m_hist.get(); }
    auto iphi() { return m_iphi.get(); }
    auto const* c_iphi() const { return m_iphi.get(); }

    auto xlToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_xl = cms::alpakatools::createDeviceView<float>(m_xl.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_xl, nHits());
      return ret;
    }
    auto ylToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_yl = cms::alpakatools::createDeviceView<float>(m_yl.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_yl, nHits());
      return ret;
    }
    auto xerrToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_xerr = cms::alpakatools::createDeviceView<float>(m_xerr.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_xerr, nHits());
      return ret;
    }
    auto yerrToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_yerr = cms::alpakatools::createDeviceView<float>(m_yerr.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_yerr, nHits());
      return ret;
    }
    auto xgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_xg = cms::alpakatools::createDeviceView<float>(m_xg.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_xg, nHits());
      return ret;
    }
    auto ygToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_yg = cms::alpakatools::createDeviceView<float>(m_yg.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_yg, nHits());
      return ret;
    }
    auto zgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_zg = cms::alpakatools::createDeviceView<float>(m_zg.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_zg, nHits());
      return ret;
    }
    auto rgToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<float>(nHits());
      auto view_ret = cms::alpakatools::createHostView<float>(ret.get(), nHits());
      auto view_m_rg = cms::alpakatools::createDeviceView<float>(m_rg.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_rg, nHits());
      return ret;
    }
    auto chargeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<int32_t>(nHits());
      auto view_ret = cms::alpakatools::createHostView<int32_t>(ret.get(), nHits());
      auto view_m_charge = cms::alpakatools::createDeviceView<int32_t>(m_charge.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_charge, nHits());
      return ret;
    }
    auto xsizeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<int16_t>(nHits());
      auto view_ret = cms::alpakatools::createHostView<int16_t>(ret.get(), nHits());
      auto view_m_xsize = cms::alpakatools::createDeviceView<int16_t>(m_xsize.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_xsize, nHits());
      return ret;
    }
    auto ysizeToHostAsync(Queue& queue) const {
      auto ret = cms::alpakatools::make_host_unique<int16_t>(nHits());
      auto view_ret = cms::alpakatools::createHostView<int16_t>(ret.get(), nHits());
      auto view_m_ysize = cms::alpakatools::createDeviceView<int16_t>(m_ysize.get(), nHits());
      alpaka::memcpy(queue, view_ret, view_m_ysize, nHits());
      return ret;
    }
#ifdef TODO
    // only the local coord and detector index
    cms::cuda::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
    cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
#endif
    auto const* xl() const { return m_xl.get(); }
    auto const* yl() const { return m_yl.get(); }
    auto const* xerr() const { return m_xerr.get(); }
    auto const* yerr() const { return m_yerr.get(); }
    auto const* xg() const { return m_xg.get(); }
    auto const* yg() const { return m_yg.get(); }
    auto const* zg() const { return m_zg.get(); }
    auto const* rg() const { return m_rg.get(); }
    auto const* charge() const { return m_charge.get(); }
    auto const* xsize() const { return m_xsize.get(); }
    auto const* ysize() const { return m_ysize.get(); }

  private:
    uint32_t m_nHits;

    // NON-OWNING DEVICE POINTERS
    // m_hitsModuleStart data is already owned by SiPixelClusterAlpaka, let's not abuse of shared_ptr!!
    uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

    // OWNING DEVICE POINTERS
    // local coord
    cms::alpakatools::device::unique_ptr<float> m_xl;
    cms::alpakatools::device::unique_ptr<float> m_yl;
    cms::alpakatools::device::unique_ptr<float> m_xerr;
    cms::alpakatools::device::unique_ptr<float> m_yerr;

    // global coord
    cms::alpakatools::device::unique_ptr<float> m_xg;
    cms::alpakatools::device::unique_ptr<float> m_yg;
    cms::alpakatools::device::unique_ptr<float> m_zg;
    cms::alpakatools::device::unique_ptr<float> m_rg;
    cms::alpakatools::device::unique_ptr<int16_t> m_iphi;

    // cluster properties
    cms::alpakatools::device::unique_ptr<int32_t> m_charge;
    cms::alpakatools::device::unique_ptr<int16_t> m_xsize;
    cms::alpakatools::device::unique_ptr<int16_t> m_ysize;
    cms::alpakatools::device::unique_ptr<uint16_t> m_detInd;

    cms::alpakatools::device::unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_averageGeometry;

    // needed as kernel params...
    cms::alpakatools::device::unique_ptr<uint32_t> m_hitsLayerStart;
    cms::alpakatools::device::unique_ptr<Hist> m_hist;

    // This is a SoA view which itself gathers non-owning pointers to the data owned above (in TrackingRecHit2DAlpaka instance).
    // This is used to access and modify data on GPU in a SoA format (TrackingRecHit2DSOAView),
    // while the data itself is owned here in the TrackingRecHit2DAlpaka instance.
    cms::alpakatools::device::unique_ptr<TrackingRecHit2DSOAView> m_view;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
