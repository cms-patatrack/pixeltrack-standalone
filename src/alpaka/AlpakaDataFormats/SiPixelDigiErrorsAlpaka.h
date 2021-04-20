#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/PixelErrors.h"
#include "AlpakaCore/SimpleVector.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class SiPixelDigiErrorsAlpaka {
public:
  SiPixelDigiErrorsAlpaka() = default;
  explicit SiPixelDigiErrorsAlpaka(size_t maxFedWords, PixelFormatterErrors errors)
    : data_d{cms::alpakatools::allocDeviceBuf<PixelErrorCompact>(maxFedWords)},
    error_d{cms::alpakatools::allocDeviceBuf<cms::alpakatools::SimpleVector<PixelErrorCompact>>(1u)},
      error_h{cms::alpakatools::allocHostBuf<cms::alpakatools::SimpleVector<PixelErrorCompact>>(1u)},
	formatterErrors_h{std::move(errors)}
	 {
	  auto perror_h = alpaka::getPtrNative(error_h);
	  perror_h->construct(maxFedWords, alpaka::getPtrNative(data_d));
	  assert(perror_h->empty());
	  assert(perror_h->capacity() == static_cast<int>(maxFedWords));

    // TO DO: nothing really async in here for now... Pass the queue in constructor argument instead, and don't wait anymore!
    Queue queue(device);
    alpaka::memcpy(queue, error_d, error_h, 1u);
    alpaka::wait(queue);
  }
  ~SiPixelDigiErrorsAlpaka() = default;

  SiPixelDigiErrorsAlpaka(const SiPixelDigiErrorsAlpaka&) = delete;
  SiPixelDigiErrorsAlpaka& operator=(const SiPixelDigiErrorsAlpaka&) = delete;
  SiPixelDigiErrorsAlpaka(SiPixelDigiErrorsAlpaka&&) = default;
  SiPixelDigiErrorsAlpaka& operator=(SiPixelDigiErrorsAlpaka&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::alpakatools::SimpleVector<PixelErrorCompact>* error() { return alpaka::getPtrNative(error_d); }
  cms::alpakatools::SimpleVector<PixelErrorCompact> const* error() const { return alpaka::getPtrNative(error_d); }
  cms::alpakatools::SimpleVector<PixelErrorCompact> const* c_error() const { return alpaka::getPtrNative(error_d); }

#ifdef TODO
  using HostDataError =
      std::pair<cms::alpakatools::SimpleVector<PixelErrorCompact>, AlpakaHostBuf<PixelErrorCompact>>;
  HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

  void copyErrorToHostAsync(cudaStream_t stream);
#endif

private:
  AlpakaDeviceBuf<PixelErrorCompact> data_d;
  AlpakaDeviceBuf<cms::alpakatools::SimpleVector<PixelErrorCompact>> error_d;
  AlpakaHostBuf<cms::alpakatools::SimpleVector<PixelErrorCompact>> error_h;
  PixelFormatterErrors formatterErrors_h;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
