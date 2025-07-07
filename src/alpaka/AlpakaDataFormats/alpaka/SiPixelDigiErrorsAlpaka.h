#ifndef AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
#define AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h

#include <utility>

#include "AlpakaCore/SimpleVector.h"
#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "DataFormats/PixelErrors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelDigiErrorsAlpaka {
  public:
    SiPixelDigiErrorsAlpaka() = delete;  // alpaka buffers are not default-constructible
    explicit SiPixelDigiErrorsAlpaka(Queue& queue, size_t maxFedWords, PixelFormatterErrors errors)
        : data_d{cms::alpakatools::make_device_buffer<PixelErrorCompact[]>(queue, maxFedWords)},
          error_d{cms::alpakatools::make_device_buffer<cms::alpakatools::SimpleVector<PixelErrorCompact>>(queue)},
          error_h{cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<PixelErrorCompact>>(queue)},
          formatterErrors_h{std::move(errors)} {
      error_h->construct(maxFedWords, data_d.data());
      ALPAKA_ASSERT_ACC(error_h->empty());
      ALPAKA_ASSERT_ACC(error_h->capacity() == static_cast<int>(maxFedWords));

      alpaka::memcpy(queue, error_d, error_h);
    }
    ~SiPixelDigiErrorsAlpaka() = default;

    SiPixelDigiErrorsAlpaka(const SiPixelDigiErrorsAlpaka&) = delete;
    SiPixelDigiErrorsAlpaka& operator=(const SiPixelDigiErrorsAlpaka&) = delete;
    SiPixelDigiErrorsAlpaka(SiPixelDigiErrorsAlpaka&&) = default;
    SiPixelDigiErrorsAlpaka& operator=(SiPixelDigiErrorsAlpaka&&) = default;

    const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

    cms::alpakatools::SimpleVector<PixelErrorCompact>* error() { return error_d.data(); }
    cms::alpakatools::SimpleVector<PixelErrorCompact> const* error() const { return error_d.data(); }
    cms::alpakatools::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.data(); }

#ifdef TODO
    using HostDataError =
        std::pair<cms::alpakatools::SimpleVector<PixelErrorCompact>, cms::alpakatools::host_buffer<PixelErrorCompact>>;
    HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

    void copyErrorToHostAsync(cudaStream_t stream);
#endif

  private:
    cms::alpakatools::device_buffer<Device, PixelErrorCompact[]> data_d;
    cms::alpakatools::device_buffer<Device, cms::alpakatools::SimpleVector<PixelErrorCompact>> error_d;
    cms::alpakatools::host_buffer<cms::alpakatools::SimpleVector<PixelErrorCompact>> error_h;
    PixelFormatterErrors formatterErrors_h;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
