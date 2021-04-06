#include "CUDADataFormats/SiPixelDigiErrorsCUDA.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/memsetAsync.h"

#include <cassert>

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, hipStream_t stream)
    : formatterErrors_h(std::move(errors)) {
  error_d = cms::hip::make_device_unique<cms::hip::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cms::hip::make_device_unique<PixelErrorCompact[]>(maxFedWords, stream);

  cms::hip::memsetAsync(data_d, 0x00, maxFedWords, stream);

  error_h = cms::hip::make_host_unique<cms::hip::SimpleVector<PixelErrorCompact>>(stream);
  cms::hip::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cms::hip::copyAsync(error_d, error_h, stream);
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(hipStream_t stream) { cms::hip::copyAsync(error_h, error_d, stream); }

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(hipStream_t stream) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cms::hip::make_host_unique<PixelErrorCompact[]>(error_h->capacity(), stream);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cms::hip::copyAsync(data, data_d, error_h->size(), stream);
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(err, std::move(data));
}
