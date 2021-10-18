#ifndef AlpakaDataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define AlpakaDataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/PixelErrors.h"
#include "AlpakaCore/SimpleVector.h"

#include "AlpakaCore/device_unique_ptr.h"
#include "AlpakaCore/host_unique_ptr.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelDigiErrorsAlpaka {
  public:
    SiPixelDigiErrorsAlpaka() = default;
    explicit SiPixelDigiErrorsAlpaka(size_t maxFedWords, PixelFormatterErrors errors)
        : data_d{cms::alpakatools::make_device_unique<PixelErrorCompact>(maxFedWords)},
          error_d{cms::alpakatools::make_device_unique<cms::alpakatools::SimpleVector<PixelErrorCompact>>(1u)},
          error_h{cms::alpakatools::make_host_unique<cms::alpakatools::SimpleVector<PixelErrorCompact>>(1u)},
          formatterErrors_h{std::move(errors)} {
      auto perror_h = error_h.get();
      perror_h->construct(maxFedWords, data_d.get());
      assert(perror_h->empty());
      assert(perror_h->capacity() == static_cast<int>(maxFedWords));

      // TO DO: nothing really async in here for now... Pass the queue in constructor argument instead, and don't wait anymore!
      Queue queue{device};
      auto error_h_view =
          cms::alpakatools::createHostView<cms::alpakatools::SimpleVector<PixelErrorCompact>>(error_h.get(), 1u);
      auto error_d_view =
          cms::alpakatools::createDeviceView<cms::alpakatools::SimpleVector<PixelErrorCompact>>(error_d.get(), 1u);
      alpaka::memcpy(queue, error_d_view, error_h_view, 1u);
      alpaka::wait(queue);
    }
    ~SiPixelDigiErrorsAlpaka() = default;

    SiPixelDigiErrorsAlpaka(const SiPixelDigiErrorsAlpaka&) = delete;
    SiPixelDigiErrorsAlpaka& operator=(const SiPixelDigiErrorsAlpaka&) = delete;
    SiPixelDigiErrorsAlpaka(SiPixelDigiErrorsAlpaka&&) = default;
    SiPixelDigiErrorsAlpaka& operator=(SiPixelDigiErrorsAlpaka&&) = default;

    const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

    cms::alpakatools::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
    cms::alpakatools::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
    cms::alpakatools::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

#ifdef TODO
    using HostDataError =
        std::pair<cms::alpakatools::SimpleVector<PixelErrorCompact>, AlpakaHostBuf<PixelErrorCompact>>;
    HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

    void copyErrorToHostAsync(cudaStream_t stream);
#endif

  private:
    cms::alpakatools::device::unique_ptr<PixelErrorCompact> data_d;
    cms::alpakatools::device::unique_ptr<cms::alpakatools::SimpleVector<PixelErrorCompact>> error_d;
    cms::alpakatools::host::unique_ptr<cms::alpakatools::SimpleVector<PixelErrorCompact>> error_h;
    PixelFormatterErrors formatterErrors_h;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
