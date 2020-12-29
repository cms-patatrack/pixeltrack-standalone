#include "CUDADataFormats/SiPixelDigisSoA.h"

SiPixelDigisSoA::SiPixelDigisSoA(size_t maxFedWords) {
  xx_d = std::make_unique<uint16_t[]>(maxFedWords);
  yy_d = std::make_unique<uint16_t[]>(maxFedWords);
  adc_d = std::make_unique<uint16_t[]>(maxFedWords);
  moduleInd_d = std::make_unique<uint16_t[]>(maxFedWords);
  clus_d = std::make_unique<int32_t[]>(maxFedWords);

  pdigi_d = std::make_unique<uint32_t[]>(maxFedWords);
  rawIdArr_d = std::make_unique<uint32_t[]>(maxFedWords);

  auto view = std::make_unique<DeviceConstView>();
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = std::move(view);
}
