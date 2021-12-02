#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "CUDACore/cudaCompat.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "DataFormats/SoAStore.h"
#include "DataFormats/SoAView.h"

class SiPixelDigisCUDA {
public:
  generate_SoA_store(DeviceOnlyLayoutTemplate,
    /* These are consumed by downstream device code                                                   */
    SoA_column(uint16_t, xx),         /* local coordinates of each pixel                              */
    SoA_column(uint16_t, yy),         /*                                                              */
    SoA_column(uint16_t, moduleInd)   /* module id of each pixel                                      */
  );
  
  using DeviceOnlyLayout = DeviceOnlyLayoutTemplate<>;
  
  generate_SoA_store(HostDeviceLayoutTemplate,
    /* These are also transferred to host (see HostDataView) */
    SoA_column(uint16_t, adc),        /* ADC of each pixel                                            */
    SoA_column(int32_t, clus),        /* cluster id of each pixel                                     */
    /* These are for CPU output; should we (eventually) place them to a                               */
    /* separate product?                                                                              */
    SoA_column(uint32_t, pdigi),     /* packed digi (row, col, adc) of each pixel                     */
    SoA_column(uint32_t, rawIdArr)   /* DetId of each pixel                                           */
  );
  
  using HostDeviceLayout = HostDeviceLayoutTemplate<>;
  
  generate_SoA_view(HostDeviceViewTemplate,
    SoA_view_store_list(
      SoA_view_store(HostDeviceLayout, hostDevice)
    ),
    SoA_view_value_list(
      SoA_view_value(hostDevice, adc),      /* ADC of each pixel                                      */
      SoA_view_value(hostDevice, clus),     /* cluster id of each pixel                               */
      SoA_view_value(hostDevice, pdigi),    /* packed digi (row, col, adc) of each pixel              */
      SoA_view_value(hostDevice, rawIdArr)  /* DetId of each pixel                                    */
    )    
  );
  
  using HostDeviceView = HostDeviceViewTemplate<>;
  
  generate_SoA_view(DeviceFullViewTemplate,
    SoA_view_store_list(
      SoA_view_store(DeviceOnlyLayout, deviceOnly),
      SoA_view_store(HostDeviceLayout, hostDevice)
    ),
    SoA_view_value_list(
      SoA_view_value(deviceOnly, xx),       /* local coordinates of each pixel                        */
      SoA_view_value(deviceOnly, yy),       /*                                                        */
      SoA_view_value(deviceOnly, moduleInd),/* module id of each pixel                                */
      SoA_view_value(hostDevice, adc),      /* ADC of each pixel                                      */
      SoA_view_value(hostDevice, clus),     /* cluster id of each pixel                               */
      SoA_view_value(hostDevice, pdigi),    /* packed digi (row, col, adc) of each pixel              */
      SoA_view_value(hostDevice, rawIdArr)  /* DetId of each pixel                                    */
    )    
  );
  
  using DeviceFullView = DeviceFullViewTemplate<>;

  /* Device pixel view: this is a second generation view (view from view) */
  generate_SoA_const_view(DevicePixelConstViewTemplate,
    /* We get out data from the DeviceFullStore */
    SoA_view_store_list(
      SoA_view_store(DeviceFullView, deviceFullView)
    ),
    /* These are consumed by downstream device code                                                   */
    SoA_view_value_list(
      SoA_view_value(deviceFullView, xx),    /* local coordinates of each pixel                       */
      SoA_view_value(deviceFullView, yy),    /*                                                       */
      SoA_view_value(deviceFullView, moduleInd),  /* module id of each pixel                          */
      SoA_view_value(deviceFullView, adc),  /* ADC of each pixel                                      */
      SoA_view_value(deviceFullView, clus) /* cluster id of each pixel                                */
    )
  );
  
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

  class HostStoreAndBuffer {
    friend SiPixelDigisCUDA;
  public:
    HostStoreAndBuffer();
    const SiPixelDigisCUDA::HostDeviceLayout store() { return hostLayout_; }
    void reset();
  private:
    HostStoreAndBuffer(size_t maxFedWords, cudaStream_t stream);
    cms::cuda::host::unique_ptr<std::byte[]> data_h;
    HostDeviceLayout hostLayout_;
    HostDeviceView hostView_;
    
  };
  HostStoreAndBuffer dataToHostAsync(cudaStream_t stream) const;

   // Special copy for validation
   cms::cuda::host::unique_ptr<uint16_t[]> adcToHostAsync(cudaStream_t stream) const;

  const DevicePixelConstView& pixelConstView() const { return devicePixelConstView_; }

private:
  // These are consumed by downstream device code
  cms::cuda::device::unique_ptr<std::byte[]> data_d;      // Single SoA storage
  DeviceOnlyLayout deviceOnlyLayout_d;
  HostDeviceLayout hostDeviceLayout_d;
  DeviceFullView deviceFullView_;
  DevicePixelConstView devicePixelConstView_;
  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h