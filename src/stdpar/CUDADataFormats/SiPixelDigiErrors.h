#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrors_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrors_h

#include <memory>

#include "CUDACore/SimpleVector.h"
#include "DataFormats/PixelErrors.h"

class SiPixelDigiErrors {
public:
  SiPixelDigiErrors() = default;
  explicit SiPixelDigiErrors(size_t maxFedWords, PixelFormatterErrors errors);
  ~SiPixelDigiErrors() = default;

  SiPixelDigiErrors(const SiPixelDigiErrors&) = delete;
  SiPixelDigiErrors& operator=(const SiPixelDigiErrors&) = delete;
  SiPixelDigiErrors(SiPixelDigiErrors&&) = default;
  SiPixelDigiErrors& operator=(SiPixelDigiErrors&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::cuda::SimpleVector<PixelErrorCompact>* error() { return errors_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* error() const { return errors_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* c_error() const { return errors_d.get(); }

private:
  size_t maxFedWords_;
  PixelFormatterErrors formatterErrors_h;
  std::unique_ptr<cms::cuda::SimpleVector<PixelErrorCompact>> errors_d;
  std::unique_ptr<PixelErrorCompact[]> data_d;
};

#endif
