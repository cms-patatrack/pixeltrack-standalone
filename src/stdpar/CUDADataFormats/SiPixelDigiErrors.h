#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrors_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrors_h

#include <memory>
#include <vector>

#include "DataFormats/PixelErrors.h"

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors)
      : formatterErrors_h(std::move(errors)) {}
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  std::vector<PixelErrorCompact>* error() { return &errors; }
  std::vector<PixelErrorCompact> const* error() const { return &errors; }
  std::vector<PixelErrorCompact> const* c_error() const { return &errors; }

private:
  std::vector<PixelErrorCompact> errors;
  size_t maxFedWords_;

  PixelFormatterErrors formatterErrors_h;
};

#endif
