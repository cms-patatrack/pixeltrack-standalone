#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsSoA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsSoA_h

#include <memory>

#include "CUDACore/SimpleVector.h"
#include "DataFormats/PixelErrors.h"

class SiPixelDigiErrorsSoA {
public:
  SiPixelDigiErrorsSoA() = default;
  explicit SiPixelDigiErrorsSoA(size_t maxFedWords, PixelFormatterErrors errors);
  ~SiPixelDigiErrorsSoA() = default;

  SiPixelDigiErrorsSoA(const SiPixelDigiErrorsSoA&) = delete;
  SiPixelDigiErrorsSoA& operator=(const SiPixelDigiErrorsSoA&) = delete;
  SiPixelDigiErrorsSoA(SiPixelDigiErrorsSoA&&) = default;
  SiPixelDigiErrorsSoA& operator=(SiPixelDigiErrorsSoA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::cuda::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

private:
  std::unique_ptr<PixelErrorCompact[]> data_d;
  std::unique_ptr<cms::cuda::SimpleVector<PixelErrorCompact>> error_d;
  PixelFormatterErrors formatterErrors_h;
};

#endif
