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
  generate_SoA_store(DeviceOnlyStore,
    /* These are consumed by downstream device code                                                   */
    SoA_column(uint16_t, xx),         /* local coordinates of each pixel                              */
    SoA_column(uint16_t, yy),         /*                                                              */
    SoA_column(uint16_t, moduleInd)   /* module id of each pixel                                      */
  );
  
  generate_SoA_store(HostDeviceStore,
    /* These are also transferred to host (see HostDataView) */
    SoA_column(uint16_t, adc),        /* ADC of each pixel                                            */
    SoA_column(int32_t, clus),        /* cluster id of each pixel                                     */
    /* These are for CPU output; should we (eventually) place them to a                               */
    /* separate product?                                                                              */
    SoA_column(uint32_t, pdigi),     /* packed digi (row, col, adc) of each pixel                     */
    SoA_column(uint32_t, rawIdArr)   /* DetId of each pixel                                           */
  );
  
  generate_SoA_view(DeviceFullView,
    SoA_view_store_list(
      SoA_view_store(DeviceOnlyStore, deviceOnly),
      SoA_view_store(HostDeviceStore, hostDevice)
    ),
    SoA_view_value_list(
      SoA_view_value(deviceOnly, xx, xx),    /* local coordinates of each pixel                       */
      SoA_view_value(deviceOnly, yy, yy),    /*                                                       */
      SoA_view_value(deviceOnly, moduleInd, moduleInd),  /* module id of each pixel                   */
      SoA_view_value(hostDevice, adc, adc),  /* ADC of each pixel                                     */
      SoA_view_value(hostDevice, clus, clus),/* cluster id of each pixel                              */
      SoA_view_value(hostDevice, pdigi, pdigi), /* packed digi (row, col, adc) of each pixel          */
      SoA_view_value(hostDevice, rawIdArr, rawIdArr)  /* DetId of each pixel                          */
      /* TODO: simple, no rename interface */
    )    
  );

  /* Device pixel view: this is a second generation view (view from view) */
  generate_SoA_const_view(DevicePixelConstView,
    /* We get out data from the DeviceFullStore */
    SoA_view_store_list(
      SoA_view_store(DeviceFullView, deviceFullView)
    ),
    /* These are consumed by downstream device code                                                   */
    SoA_view_value_list(
      SoA_view_value(deviceFullView, xx, xx),    /* local coordinates of each pixel                   */
      SoA_view_value(deviceFullView, yy, yy),    /*                                                   */
      SoA_view_value(deviceFullView, moduleInd, moduleInd),  /* module id of each pixel               */
      SoA_view_value(deviceFullView, adc, adc),  /* ADC of each pixel                                 */
      SoA_view_value(deviceFullView, clus, clus) /* cluster id of each pixel                          */
    )
  );

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
    const SiPixelDigisCUDA::HostDeviceStore store() { return hostStore_; }
    void reset();
  private:
    HostStoreAndBuffer(size_t maxFedWords, cudaStream_t stream);
    cms::cuda::host::unique_ptr<std::byte[]> data_h;
    HostDeviceStore hostStore_;
  };
  HostStoreAndBuffer dataToHostAsync(cudaStream_t stream) const;

   // Special copy for validation
   cms::cuda::host::unique_ptr<uint16_t[]> adcToHostAsync(cudaStream_t stream) const;

  const DevicePixelConstView& pixelConstView() const { return devicePixelConstView_; }

private:
  // These are consumed by downstream device code
  cms::cuda::device::unique_ptr<std::byte[]> data_d;      // Single SoA storage
  DeviceOnlyStore deviceOnlyStore_d;
  HostDeviceStore hostDeviceStore_d;
  DeviceFullView deviceFullView_;
  DevicePixelConstView devicePixelConstView_;
  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h