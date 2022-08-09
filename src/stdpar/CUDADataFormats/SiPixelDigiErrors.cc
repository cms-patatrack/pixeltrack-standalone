#include <cassert>
#include <memory>

#include "CUDACore/SimpleVector.h"
#include "CUDADataFormats/SiPixelDigiErrors.h"

SiPixelDigiErrors::SiPixelDigiErrors(size_t maxFedWords, PixelFormatterErrors errors)
    : maxFedWords_{maxFedWords},
      formatterErrors_h{std::move(errors)},
      errors_d{std::make_unique<cms::cuda::SimpleVector<PixelErrorCompact>>()},
      data_d{std::make_unique<PixelErrorCompact[]>(maxFedWords)} {
  cms::cuda::make_SimpleVector(errors_d.get(), maxFedWords_, data_d.get());
  assert(errors_d->empty());
  assert(errors_d->capacity() == static_cast<int>(maxFedWords));
}
