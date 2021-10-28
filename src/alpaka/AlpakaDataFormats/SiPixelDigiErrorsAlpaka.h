#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/PixelErrors.h"
#include "AlpakaCore/SimpleVector.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelDigiErrorsAlpaka {
  public:
    SiPixelDigiErrorsAlpaka() = default;
    explicit SiPixelDigiErrorsAlpaka(Device const& device, size_t maxFedWords, PixelFormatterErrors errors, Queue& queue)
        : data_d{cms::alpakatools::allocDeviceBuf<PixelErrorCompact>(device, maxFedWords)},
          error_d{cms::alpakatools::allocDeviceBuf<::cms::alpakatools::SimpleVector<PixelErrorCompact>>(device, 1u)},
          error_h{::cms::alpakatools::allocHostBuf<::cms::alpakatools::SimpleVector<PixelErrorCompact>>(1u)},
          formatterErrors_h{std::move(errors)} {
      auto perror_h = alpaka::getPtrNative(error_h);
      perror_h->construct(maxFedWords, alpaka::getPtrNative(data_d));
      ALPAKA_ASSERT_OFFLOAD(perror_h->empty());
      ALPAKA_ASSERT_OFFLOAD(perror_h->capacity() == static_cast<int>(maxFedWords));

      alpaka::memcpy(queue, error_d, error_h, 1u);
    }
    ~SiPixelDigiErrorsAlpaka() = default;

    SiPixelDigiErrorsAlpaka(const SiPixelDigiErrorsAlpaka&) = delete;
    SiPixelDigiErrorsAlpaka& operator=(const SiPixelDigiErrorsAlpaka&) = delete;
    SiPixelDigiErrorsAlpaka(SiPixelDigiErrorsAlpaka&&) = default;
    SiPixelDigiErrorsAlpaka& operator=(SiPixelDigiErrorsAlpaka&&) = default;

    const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

    ::cms::alpakatools::SimpleVector<PixelErrorCompact>* error() { return alpaka::getPtrNative(error_d); }
    ::cms::alpakatools::SimpleVector<PixelErrorCompact> const* error() const { return alpaka::getPtrNative(error_d); }
    ::cms::alpakatools::SimpleVector<PixelErrorCompact> const* c_error() const { return alpaka::getPtrNative(error_d); }

#ifdef TODO
    using HostDataError =
        std::pair<::cms::alpakatools::SimpleVector<PixelErrorCompact>, AlpakaHostBuf<PixelErrorCompact>>;
    HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

    void copyErrorToHostAsync(cudaStream_t stream);
#endif

  private:
    AlpakaDeviceBuf<PixelErrorCompact> data_d;
    AlpakaDeviceBuf<::cms::alpakatools::SimpleVector<PixelErrorCompact>> error_d;
    AlpakaHostBuf<::cms::alpakatools::SimpleVector<PixelErrorCompact>> error_h;
    PixelFormatterErrors formatterErrors_h;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
