#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "CUDACore/cudaCompat.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

class SiPixelDigisCUDA {
public:
  GENERATE_SOA_LAYOUT(
      DeviceOnlyLayoutTemplate,
      /* These are consumed by downstream device code                                                   */
      SOA_COLUMN(uint16_t, xx),       /* local coordinates of each pixel                              */
      SOA_COLUMN(uint16_t, yy),       /*                                                              */
      SOA_COLUMN(uint16_t, moduleInd) /* module id of each pixel                                      */
  )

  using DeviceOnlyLayout = DeviceOnlyLayoutTemplate<>;

  GENERATE_SOA_LAYOUT(
      HostDeviceLayoutTemplate,
      /* These are also transferred to host (see HostDataView) */
      SOA_COLUMN(uint16_t, adc), /* ADC of each pixel                                            */
      SOA_COLUMN(int32_t, clus), /* cluster id of each pixel                                     */
      /* These are for CPU output; should we (eventually) place them to a                               */
      /* separate product?                                                                              */
      SOA_COLUMN(uint32_t, pdigi),   /* packed digi (row, col, adc) of each pixel                     */
      SOA_COLUMN(uint32_t, rawIdArr) /* DetId of each pixel                                           */
  )

  using HostDeviceLayout = HostDeviceLayoutTemplate<>;

  GENERATE_SOA_VIEW(HostDeviceViewTemplate,
                    SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(HostDeviceLayout, hostDevice)),
                    SOA_VIEW_VALUE_LIST(
                        SOA_VIEW_VALUE(hostDevice, adc),   /* ADC of each pixel                                      */
                        SOA_VIEW_VALUE(hostDevice, clus),  /* cluster id of each pixel                               */
                        SOA_VIEW_VALUE(hostDevice, pdigi), /* packed digi (row, col, adc) of each pixel              */
                        SOA_VIEW_VALUE(hostDevice,
                                       rawIdArr) /* DetId of each pixel                                    */
                        ))

  using HostDeviceView = HostDeviceViewTemplate<>;

  GENERATE_SOA_VIEW(
      DeviceFullViewTemplate,
      SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(DeviceOnlyLayout, deviceOnly),
                           SOA_VIEW_LAYOUT(HostDeviceLayout, hostDevice)),
      SOA_VIEW_VALUE_LIST(
          SOA_VIEW_VALUE(deviceOnly, xx),        /* local coordinates of each pixel                        */
          SOA_VIEW_VALUE(deviceOnly, yy),        /*                                                        */
          SOA_VIEW_VALUE(deviceOnly, moduleInd), /* module id of each pixel                                */
          SOA_VIEW_VALUE(hostDevice, adc),       /* ADC of each pixel                                      */
          SOA_VIEW_VALUE(hostDevice, clus),      /* cluster id of each pixel                               */
          SOA_VIEW_VALUE(hostDevice, pdigi),     /* packed digi (row, col, adc) of each pixel              */
          SOA_VIEW_VALUE(hostDevice, rawIdArr)   /* DetId of each pixel                                    */
          ))

  using DeviceFullView = DeviceFullViewTemplate<>;

  /* Device pixel view: this is a second generation view (view from view) */
  GENERATE_SOA_CONST_VIEW(
      DevicePixelConstViewTemplate,
      /* We get out data from the DeviceFullView */
      SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(DeviceFullView, deviceFullView)),
      /* These are consumed by downstream device code                                                   */
      SOA_VIEW_VALUE_LIST(
          SOA_VIEW_VALUE(deviceFullView, xx),        /* local coordinates of each pixel                       */
          SOA_VIEW_VALUE(deviceFullView, yy),        /*                                                       */
          SOA_VIEW_VALUE(deviceFullView, moduleInd), /* module id of each pixel                          */
          SOA_VIEW_VALUE(deviceFullView, adc),       /* ADC of each pixel                                      */
          SOA_VIEW_VALUE(deviceFullView, clus)       /* cluster id of each pixel                                */
          ))

  using DevicePixelConstView = DevicePixelConstViewTemplate<>;

  explicit SiPixelDigisCUDA();
  explicit SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA &operator=(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  uint16_t *xx() { return deviceFullView_.xx(); }
  uint16_t *yy() { return deviceFullView_.yy(); }
  uint16_t *adc() { return deviceFullView_.adc(); }
  uint16_t *moduleInd() { return deviceFullView_.moduleInd(); }
  int32_t *clus() { return deviceFullView_.clus(); }
  uint32_t *pdigi() { return deviceFullView_.pdigi(); }
  uint32_t *rawIdArr() { return deviceFullView_.rawIdArr(); }

  uint16_t const *xx() const { return deviceFullView_.xx(); }
  uint16_t const *yy() const { return deviceFullView_.yy(); }
  uint16_t const *adc() const { return deviceFullView_.adc(); }
  uint16_t const *moduleInd() const { return deviceFullView_.moduleInd(); }
  int32_t const *clus() const { return deviceFullView_.clus(); }
  uint32_t const *pdigi() const { return deviceFullView_.pdigi(); }
  uint32_t const *rawIdArr() const { return deviceFullView_.rawIdArr(); }

  class HostStore {
    friend SiPixelDigisCUDA;

  public:
    HostStore();
    const SiPixelDigisCUDA::HostDeviceView view() { return hostView_; }
    void reset();

  private:
    HostStore(size_t maxFedWords, cudaStream_t stream);
    cms::cuda::host::unique_ptr<std::byte[]> data_h;
    HostDeviceLayout hostLayout_;
    HostDeviceView hostView_;
  };
  HostStore dataToHostAsync(cudaStream_t stream) const;

  // Special copy for validation
  cms::cuda::host::unique_ptr<uint16_t[]> adcToHostAsync(cudaStream_t stream) const;

  const DevicePixelConstView &pixelConstView() const { return devicePixelConstView_; }

private:
  // These are consumed by downstream device code
  cms::cuda::device::unique_ptr<std::byte[]> data_d;  // Single SoA storage
  DeviceOnlyLayout deviceOnlyLayout_d;
  HostDeviceLayout hostDeviceLayout_d;
  DeviceFullView deviceFullView_;
  DevicePixelConstView devicePixelConstView_;
  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h