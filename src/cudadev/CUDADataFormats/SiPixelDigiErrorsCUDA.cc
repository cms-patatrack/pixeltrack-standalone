#include <cassert>

#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/memsetAsync.h"
#include "CUDADataFormats/SiPixelDigiErrorsCUDA.h"

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords,
                                             SiPixelFormatterErrors errors,
                                             cms::cuda::Context const& ctx)
    : data_d(cms::cuda::make_device_unique<SiPixelErrorCompact[]>(maxFedWords, ctx)),
      error_d(cms::cuda::make_device_unique<SiPixelErrorCompactVector>(ctx)),
      error_h(cms::cuda::make_host_unique<SiPixelErrorCompactVector>(ctx)),
      formatterErrors_h(std::move(errors)) {
  cms::cuda::memsetAsync(data_d, 0x00, maxFedWords, ctx.stream());

  cms::cuda::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cms::cuda::copyAsync(error_d, error_h, ctx.stream());
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cms::cuda::Context const& ctx) {
  cms::cuda::copyAsync(error_h, error_d, ctx.stream());
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cms::cuda::Context const& ctx) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cms::cuda::make_host_unique<SiPixelErrorCompact[]>(error_h->capacity(), ctx);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cms::cuda::copyAsync(data, data_d, error_h->size(), ctx.stream());
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(err, std::move(data));
}
