#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisSoA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisSoA_h

#include "CUDACore/cudaCompat.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include <memory>

class SiPixelDigisSoA {
public:
  SiPixelDigisSoA() = default;
  explicit SiPixelDigisSoA(size_t maxFedWords);
  ~SiPixelDigisSoA() = default;

  SiPixelDigisSoA(const SiPixelDigisSoA &) = delete;
  SiPixelDigisSoA &operator=(const SiPixelDigisSoA &) = delete;
  SiPixelDigisSoA(SiPixelDigisSoA &&) = default;
  SiPixelDigisSoA &operator=(SiPixelDigisSoA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  uint16_t *xx() { return xx_d.get(); }
  uint16_t *yy() { return yy_d.get(); }
  uint16_t *adc() { return adc_d.get(); }
  uint16_t *moduleInd() { return moduleInd_d.get(); }
  int32_t *clus() { return clus_d.get(); }
  uint32_t *pdigi() { return pdigi_d.get(); }
  uint32_t *rawIdArr() { return rawIdArr_d.get(); }

  uint16_t const *xx() const { return xx_d.get(); }
  uint16_t const *yy() const { return yy_d.get(); }
  uint16_t const *adc() const { return adc_d.get(); }
  uint16_t const *moduleInd() const { return moduleInd_d.get(); }
  int32_t const *clus() const { return clus_d.get(); }
  uint32_t const *pdigi() const { return pdigi_d.get(); }
  uint32_t const *rawIdArr() const { return rawIdArr_d.get(); }

  uint16_t const *c_xx() const { return xx_d.get(); }
  uint16_t const *c_yy() const { return yy_d.get(); }
  uint16_t const *c_adc() const { return adc_d.get(); }
  uint16_t const *c_moduleInd() const { return moduleInd_d.get(); }
  int32_t const *c_clus() const { return clus_d.get(); }
  uint32_t const *c_pdigi() const { return pdigi_d.get(); }
  uint32_t const *c_rawIdArr() const { return rawIdArr_d.get(); }

  using DeviceConstView = SiPixelDigisCUDA::DeviceConstView;

  const DeviceConstView *view() const { return view_d.get(); }

private:
  // These are consumed by downstream device code
  std::unique_ptr<uint16_t[]> xx_d;         // local coordinates of each pixel
  std::unique_ptr<uint16_t[]> yy_d;         //
  std::unique_ptr<uint16_t[]> adc_d;        // ADC of each pixel
  std::unique_ptr<uint16_t[]> moduleInd_d;  // module id of each pixel
  std::unique_ptr<int32_t[]> clus_d;        // cluster id of each pixel
  std::unique_ptr<DeviceConstView> view_d;  // "me" pointer

  // These are for CPU output; should we (eventually) place them to a
  // separate product?
  std::unique_ptr<uint32_t[]> pdigi_d;
  std::unique_ptr<uint32_t[]> rawIdArr_d;

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif
