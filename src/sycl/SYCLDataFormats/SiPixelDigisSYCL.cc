#include "SYCLDataFormats/SiPixelDigisSYCL.h"

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

SiPixelDigisSYCL::SiPixelDigisSYCL(size_t maxFedWords, sycl::queue stream) {
  xx_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  adc_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::sycltools::make_device_unique<int32_t[]>(maxFedWords, stream);

  pdigi_d = cms::sycltools::make_device_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::sycltools::make_device_unique<uint32_t[]>(maxFedWords, stream);

  auto view = cms::sycltools::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cms::sycltools::make_device_unique<DeviceConstView>(stream);
  stream.memcpy(view_d.get(), view.get(), sizeof(DeviceConstView));
}

cms::sycltools::host::unique_ptr<uint16_t[]> SiPixelDigisSYCL::adcToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint16_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), adc_d.get(), nDigis() * sizeof(uint16_t));
  return ret;
}

cms::sycltools::host::unique_ptr<int32_t[]> SiPixelDigisSYCL::clusToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int32_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), clus_d.get(), nDigis() * sizeof(int32_t));
  return ret;
}

cms::sycltools::host::unique_ptr<uint32_t[]> SiPixelDigisSYCL::pdigiToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), pdigi_d.get(), nDigis() * sizeof(uint32_t));
  return ret;
}

cms::sycltools::host::unique_ptr<uint32_t[]> SiPixelDigisSYCL::rawIdArrToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), rawIdArr_d.get(), nDigis() * sizeof(uint32_t));
  return ret;
}
