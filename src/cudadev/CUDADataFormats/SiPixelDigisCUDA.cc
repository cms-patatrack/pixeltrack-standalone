#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cms::cuda::Context const& ctx)
    : xx_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, ctx)),
      yy_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, ctx)),
      adc_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, ctx)),
      moduleInd_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, ctx)),
      clus_d(cms::cuda::make_device_unique<int32_t[]>(maxFedWords, ctx)),
      view_d(cms::cuda::make_device_unique<DeviceConstView>(ctx)),
      pdigi_d(cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, ctx)),
      rawIdArr_d(cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, ctx)) {
  auto view = cms::cuda::make_host_unique<DeviceConstView>(ctx);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  cms::cuda::copyAsync(view_d, view, ctx.stream());
}

cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), ctx);
  cms::cuda::copyAsync(ret, adc_d, nDigis(), ctx.stream());
  return ret;
}

cms::cuda::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), ctx);
  cms::cuda::copyAsync(ret, clus_d, nDigis(), ctx.stream());
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), ctx);
  cms::cuda::copyAsync(ret, pdigi_d, nDigis(), ctx.stream());
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cms::cuda::Context const& ctx) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), ctx);
  cms::cuda::copyAsync(ret, rawIdArr_d, nDigis(), ctx.stream());
  return ret;
}
