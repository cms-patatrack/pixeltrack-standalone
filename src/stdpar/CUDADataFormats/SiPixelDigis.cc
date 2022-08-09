#include <memory>

#include "CUDADataFormats/SiPixelDigis.h"

SiPixelDigis::SiPixelDigis(size_t maxFedWords)
    : xx_d{std::make_unique<uint16_t[]>(maxFedWords)},
      yy_d{std::make_unique<uint16_t[]>(maxFedWords)},
      moduleInd_d{std::make_unique<uint16_t[]>(maxFedWords)},
      adc_d{std::make_unique<uint16_t[]>(maxFedWords)},
      clus_d{std::make_unique<int32_t[]>(maxFedWords)},
      view_d{std::make_unique<DeviceConstView>()},
      pdigi_d{std::make_unique<uint32_t[]>(maxFedWords)},
      rawIdArr_d{std::make_unique<uint32_t[]>(maxFedWords)} {
  view_d->xx_ = xx_d.get();
  view_d->yy_ = yy_d.get();
  view_d->adc_ = adc_d.get();
  view_d->moduleInd_ = moduleInd_d.get();
  view_d->clus_ = clus_d.get();
}
